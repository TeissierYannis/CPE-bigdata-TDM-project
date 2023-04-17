import io
import json
import subprocess
import uuid

import pandas as pd
import torch
import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from celery.utils.log import get_task_logger
import concurrent.futures


from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import DetrImageProcessor, DetrForObjectDetection
from sqlalchemy import create_engine
from PIL import Image
from minio import Minio
import psycopg2
import psycopg2.extensions
import logging
from celery import Celery

load_dotenv()

app = Flask(__name__)

binary = '/app/shared/config/exifextract'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the MinIO client
minio_client = Minio(
    "minio:9000",  # Adda to .env
    access_key="minio",  # Add to .env
    secret_key="minio123",  # Add to .env
    secure=False  # Add to .env
)
# celery -A app.celery worker --loglevel=info
# Add this Celery configuration
app.config['CELERY_BROKER_URL'] = 'pyamqp://guest:guest@rabbitmq:5672//'
app.config['CELERY_RESULT_BACKEND'] = 'rpc://'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
cel_logger = get_task_logger(__name__)

engine = create_engine("postgresql://postgres:postgres@postgres:5432/raw_metadata")

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def detect_with_transformers(image):
    """
    This function detects objects in an image using the DETR (DEtection TRansformer) model by Facebook.

    Args:
    image: A string representing the path of the image to be processed.

    Returns:
    A list containing the labels of the detected objects in the image.

    Raises:
    None.
    """
    # resize image to 800x800
    cel_logger.info("Resizing image to 800x800")
    image = image.resize((800, 800))
    cel_logger.info("Detecting objects in image")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    cel_logger.info("Converting outputs to COCO API")
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]
    labels = []
    cel_logger.info("Processing results")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        labels.append(model.config.id2label[label.item()])
    cel_logger.info("Returning labels")
    return labels


def detect_with_transformers_with_timeout(image):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(detect_with_transformers, image)
        try:
            result = future.result(timeout=60)
        except concurrent.futures.TimeoutError:
            cel_logger.warning("Object detection timed out, returning an empty array")
            result = []
    return result


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def find_dominant_colors(pil_image, k=4, downsample=2, resize=(200, 200)):
    # Convert PIL Image to a NumPy array and convert to RGB
    cel_logger.info("Converting image to NumPy array")
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cel_logger.info("Resizing image")

    # Downsample the image
    image = cv2.resize(image, (image.shape[1] // downsample, image.shape[0] // downsample))

    # Resize the image if requested
    if resize is not None:
        image = cv2.resize(image, resize)

    # Flatten the image
    cel_logger.info("Flattening image")
    image_flat = image.reshape((image.shape[0] * image.shape[1], 3))

    cel_logger.info("Clustering pixels")
    # Cluster the pixels using KMeans and find percentage of image covered by each color
    clt = MiniBatchKMeans(n_clusters=k, n_init=5, batch_size=200, random_state=42)
    labels = clt.fit_predict(image_flat)

    # Count the number of pixels assigned to each cluster
    counts = np.bincount(labels)

    # Calculate the percentage of pixels assigned to each cluster
    percentages = counts / len(labels)

    # Get the dominant colors$
    cel_logger.info("Getting dominant colors")
    dominant_colors = clt.cluster_centers_

    # Convert to hexadecimal format
    cel_logger.info("Converting to hexadecimal")
    dominant_colors_hex = [rgb_to_hex(color) for color in dominant_colors]

    # Combine the dominant colors and their percentages into a array of tuples
    cel_logger.info("Combining dominant colors and percentages")
    result = list(zip(dominant_colors_hex, percentages))

    cel_logger.info("Returning dominant colors")
    return result


def find_dominant_colors_with_timeout(image):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(find_dominant_colors, image, k=4, downsample=2, resize=(200, 200))
        try:
            result = future.result(timeout=60)
        except concurrent.futures.TimeoutError:
            cel_logger.warning("Dominant color detection timed out, returning an empty array")
            result = []
    return result


def extract_metadata(image_data, image_name):
    """
    Extract metadata from images using custom exiftool script
    :return: success message
    """
    # Create a random temp folder to store the image
    cel_logger.info("Extracting metadata from image")
    temp_folder = '/app/shared/temp/' + str(uuid.uuid4())
    subprocess.call(['mkdir', '-p', temp_folder])
    cel_logger.info("Created temp folder")

    # Save the image to the temp folder
    cel_logger.info("Saving image to temp folder")
    image = Image.open(image_data)
    image.save(temp_folder + '/' + image_name, exif=image.info.get('exif'))
    cel_logger.info("Saved image to temp folder")

    # execute exiftool script to extract metadata
    cel_logger.info("Extracting metadata")
    command = [binary, temp_folder, temp_folder + '/metadata.csv']
    subprocess.call(command)
    cel_logger.info("Extracted metadata")

    # Read the metadata from the csv as pandas dataframe
    cel_logger.info("Reading metadata")
    metadata = pd.read_csv(temp_folder + '/metadata.csv')

    # Delete the temp folder
    cel_logger.info("Deleting temp folder")
    subprocess.call(['rm', '-rf', temp_folder])

    return metadata


@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    image_filename = data.get('filename')

    logging.debug({'filename': image_filename})

    try:
        # Call the process_image_task asynchronously
        task = process_image_task.apply_async(args=(image_filename,))
        return jsonify({"response": "Image processing started, task_id: " + task.id}), 202

    except Exception as e:
        return jsonify({
            "response": 'Error processing image, error: ' + str(e),
        }), 500


@celery.task(bind=True, name='app.process_image_task')
def process_image_task(self, image_filename):
    logging.debug({'task_id': self.request.id, 'filename': image_filename, 'status': 'started'})
    # Log in celery
    cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'started'})
    try:
        # Download the image from MinIO
        image_object = minio_client.get_object("images", image_filename)
        image_data = io.BytesIO(image_object.read())
        # Log in celery
        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'image downloaded'})
    except Exception as e:
        logging.error({'task_id': self.request.id, 'filename': image_filename, 'status': 'failed',
                       'error': 'Error downloading image from MinIO'})
        # Log in celery
        cel_logger.error({'task_id': self.request.id, 'filename': image_filename, 'status': 'failed', 'error': str(e)})
        return False

    try:

        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'started processing'})
        # First, extract metadata from the image
        metadata = extract_metadata(image_data, image_filename)
        metadata['filename'] = image_filename
        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'metadata extracted'})
        # Load the image using PIL
        image = Image.open(image_data)
        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'image loaded'})
        # Get labels, metadata, and dominant colors
        labels = detect_with_transformers_with_timeout(image)

        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'labels detected'})
        # Metadata extraction not shown because it requires the image path
        dominant_colors = find_dominant_colors_with_timeout(image)

        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'dominant colors detected'})

        # Add tags to the metadata dataframe
        # Convert labels as json string
        metadata.loc[0, 'tags'] = json.dumps(labels)

        # Add dominant colors to the metadata dataframe as a json string
        metadata.loc[0, 'dominant_color'] = json.dumps(dominant_colors)

        # Set all header of the dataframe to lowercase
        # Set all header of the dataframe to lowercase
        metadata.columns = metadata.columns.str.lower()
        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'metadata updated'})

        # Save the metadata to Postgres (the table is image_metadata)
        metadata.to_sql('image_metadata', engine, if_exists='append', index=False)
        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'metadata saved'})

        # Close the image
        image.close()
        cel_logger.info({'task_id': self.request.id, 'filename': image_filename, 'status': 'image closed'})

        return 'Image processed successfully'

    except Exception as e:
        # Return the error message in case of any exceptions
        cel_logger.error({'task_id': self.request.id, 'filename': image_filename, 'status': 'failed', 'error': str(e)})
        return 'Error processing image, error: ' + str(e)


if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG)
    app.run(debug=True)
