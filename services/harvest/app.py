import base64
import subprocess
import mysql.connector
import pandas as pd
import os
import torch
import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np

from flask import Flask, request, jsonify
from mysql.connector import Error
from dotenv import load_dotenv
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

load_dotenv()

app = Flask(__name__)

# Path to save images
images_path = './images'
binary = './config/exifextract'
# Path to save metadata
metadata_path = './metadata'
# Database configuration
cnf = {
    'host': os.getenv("SQL_HOST") or 'localhost',
    'user': os.getenv("SQL_USER") or 'root',
    'port': os.getenv("SQL_PORT") or 3306,
    'password': os.getenv("SQL_PASSWORD") or '',
    'database': os.getenv("SQL_DATABASE") or 'harvest'
}


def get_all_images(path):
    """Get all images from the given path.

    Args:
    param: image_path (str): path to the directory containing the images.

    Returns:
    - list: a list of full path to all the images with png or jpg extensions.
    - empty list: an empty list if an error occurred while fetching images.
    """
    try:
        # use os.walk to traverse all the subdirectories and get all images
        return [os.path.join(root, name)
                for root, dirs, files in os.walk(path)
                for name in files
                if name.endswith((".png", ".jpg"))]
    except Exception as e:
        # return an empty list and log the error message if an error occurred
        print(f"An error occurred while fetching images: {e}")
        return []


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
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    labels = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        labels.append(model.config.id2label[label.item()])
    return labels


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def find_dominant_colors(image_path, k=4, downsample=2, resize=(200, 200)):
    # Load image and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Downsample the image
    image = cv2.resize(image, (image.shape[1] // downsample, image.shape[0] // downsample))

    # Resize the image if requested
    if resize is not None:
        image = cv2.resize(image, resize)

    # Flatten the image
    image_flat = image.reshape((image.shape[0] * image.shape[1], 3))

    # Cluster the pixels using KMeans and find percentage of image covered by each color
    clt = MiniBatchKMeans(n_clusters=k, n_init=10, batch_size=100, random_state=42)
    labels = clt.fit_predict(image_flat)

    # Count the number of pixels assigned to each cluster
    counts = np.bincount(labels)

    # Calculate the percentage of pixels assigned to each cluster
    percentages = counts / len(labels)

    # Get the dominant colors
    dominant_colors = clt.cluster_centers_

    # Convert to hexadecimal format
    dominant_colors_hex = [rgb_to_hex(color) for color in dominant_colors]

    # Combine the dominant colors and their percentages into a array of tuples
    result = list(zip(dominant_colors_hex, percentages))

    return result


@app.route('/metadata/extract', methods=['GET'])
def extract_metadata():
    """
    Extract metadata from images using custom exiftool script
    :return: success message
    """
    # execute exiftool script to extract metadata
    command = [binary, images_path, metadata_path + '/metadata.csv']
    # create folder if it doesn't exist
    subprocess.call(['mkdir', '-p', metadata_path])
    subprocess.call(command)
    # return success message as response
    return jsonify({'message': 'Metadata extracted successfully'})


@app.route('/metadata/save', methods=['GET'])
def save_metadata():
    # first read metadata from csv file and convert it to pandas dataframe
    metadata = pd.read_csv(metadata_path + '/metadata.csv', error_bad_lines=False)
    # convert dataframe to dataframe with columns filename, key, value
    metadata = pd.melt(metadata, id_vars=['filename'], var_name='key', value_name='value')

    # save metadata to database (mariadb)
    conn = mysql.connector.connect(**cnf)
    # disable auto commit
    conn.autocommit = False
    cursor = conn.cursor()
    try:
        # create table if it doesn't exist
        cursor.execute('CREATE TABLE IF NOT EXISTS metadata (filename TEXT, mkey TEXT, mvalue TEXT)')
        # insert new metadata
        for index, row in metadata.iterrows():
            try:
                # check if metadata already exists
                cursor.execute('SELECT * FROM metadata WHERE filename = %s AND mkey = %s AND mvalue = %s',
                               (row['filename'], row['key'], row['value']))
                rows = cursor.fetchall()
                if not rows:
                    # insert metadata if it doesn't exist
                    cursor.execute('INSERT INTO metadata VALUES (%s, %s, %s)',
                                   (row['filename'], row['key'], row['value']))

            except Error as e:
                continue
        conn.commit()
    except Error as e:
        # print the request error
        return jsonify({'message': 'Error while saving metadata to database: {}'.format(e)})
    finally:
        cursor.close()
        conn.autocommit = True
        conn.close()

    # return success message as response
    return jsonify({'message': 'Metadata saved successfully'})


@app.route('/metadata/search', methods=['POST'])
def search_metadata():
    search_terms = request.json.get('search_terms')
    metadata_fields = request.json.get('metadata_fields')
    # search metadata in database
    conn = mysql.connector.connect(**cnf)
    cursor = conn.cursor()
    try:
        # create table if it doesn't exist
        cursor.execute('CREATE TABLE IF NOT EXISTS metadata (filename TEXT, mkey TEXT, mvalue TEXT)')
        # select all metadata
        cursor.execute('SELECT * FROM metadata')
        # get all rows
        rows = cursor.fetchall()
        # convert rows to pandas dataframe
        metadata = pd.DataFrame(rows, columns=['filename', 'key', 'value'])
        # filter metadata based on search terms
        for search_term in search_terms:
            metadata = metadata[metadata['value'].str.contains(search_term)]
        # filter metadata based on metadata fields
        metadata = metadata[metadata['key'].isin(metadata_fields)]
        # convert dataframe to json
        metadata = metadata.to_json(orient='records')
    except Error as e:
        # print the request error
        return jsonify({'message': 'Error while searching metadata in database: {}'.format(e)})
    finally:
        cursor.close()
        conn.close()

    # return metadata as response
    return metadata


@app.route('/metadata/delete', methods=['DELETE'])
def delete_metadata():
    filename = request.json.get('filename')
    # delete metadata from database
    conn = mysql.connector.connect(**cnf)
    cursor = conn.cursor()
    try:
        # create table if it doesn't exist
        cursor.execute('CREATE TABLE IF NOT EXISTS metadata (filename TEXT, mkey TEXT, mvalue TEXT)')
        # delete metadata
        cursor.execute('DELETE FROM metadata WHERE filename = %s', (filename,))
        conn.commit()
    except Error as e:
        # print the request error
        return jsonify({'message': 'Error while deleting metadata from database: {}'.format(e)})
    finally:
        cursor.close()
        conn.close()

    # return success message as response
    return jsonify({'message': 'Metadata deleted successfully'})


@app.route('/labels/extract', methods=['GET'])
def extract_labels():
    images = get_all_images(images_path)

    # dictionary to store labels
    labels = {}
    for image in images:
        # get name of image
        image_name = image.split('/')[-1]

        image = Image.open(image)
        # resize image to 416x416
        image = image.resize((416, 416))
        # get name of image
        labels[image_name] = detect_with_transformers(image)
        image.close()

    # Save labels to csv file
    with open(metadata_path + '/labels.csv', 'w') as f:
        # add header
        f.write("filename,value\n")
        for key in labels.keys():
            f.write("%s,%s\n" % (key, labels[key]))

    # return success message as response
    return jsonify({'message': 'Labels extracted successfully'})


@app.route('/labels/save', methods=['GET'])
def save_labels():
    # first read labels from csv file and convert it to pandas dataframe
    labels = pd.read_csv(metadata_path + '/labels.csv', error_bad_lines=False)
    # convert dataframe to dataframe with columns filename, key, value
    labels = pd.melt(labels, id_vars=['filename'], value_name='value')

    # save labels to database (mariadb)
    conn = mysql.connector.connect(**cnf)
    # disable auto commit
    conn.autocommit = False
    cursor = conn.cursor()
    try:
        # create table if it doesn't exist
        cursor.execute('CREATE TABLE IF NOT EXISTS metadata (filename TEXT, mkey TEXT, mvalue TEXT)')
        # insert new labels
        for index, row in labels.iterrows():
            try:
                # check if label already exists (mkey is always 'tags')
                cursor.execute('SELECT * FROM metadata WHERE filename = %s AND mkey = %s AND mvalue = %s',
                               (row['filename'], 'tags', row['value']))
                rows = cursor.fetchall()
                if not rows:
                    # insert label if it doesn't exist
                    cursor.execute('INSERT INTO metadata VALUES (%s, %s, %s)',
                                   (row['filename'], 'tags', row['value']))

            except Error as e:
                continue
        conn.commit()
    except Error as e:
        # print the request error
        return jsonify({'message': 'Error while saving labels to database: {}'.format(e)})
    finally:
        cursor.close()
        conn.autocommit = True
        conn.close()

    # return success message as response
    return jsonify({'message': 'Labels saved successfully'})


@app.route('/colors/extract', methods=['GET'])
def extract_colors():
    # Get a list of all images in the directory
    img_files = get_all_images(images_path)
    colors = {}

    # Create a progress bar to track the progress of processing all images
    for img in img_files:
        try:
            # Create a list of coroutines to extract metadata for all images
            color = find_dominant_colors(img, downsample=2, resize=(100, 100))
        except Exception as e:
            print("Error: ", e)
            continue

        if color:
            # color to string to avoid errors with quote marks
            color = str(color)
            # replace quotes by double quotes
            color = color.replace("'", '"')
            # get name of image
            image_name = img.split('/')[-1]
            # convert color string to base64
            color = base64.b64encode(color.encode('utf-8')).decode('utf-8')
            colors[image_name] = color

    # Save colors to csv file
    with open(metadata_path + '/colors.csv', 'w') as f:
        # add header
        f.write("filename,value")
        for key in colors.keys():
            # reformats the color string to be compatible with the csv format
            f.write("\n%s,%s" % (key, colors[key]))

    # return success message as response
    return jsonify({'message': 'Colors extracted successfully'})


@app.route('/colors/save', methods=['GET'])
def save_colors():
    # first read colors from csv file and convert it to pandas dataframe
    colors = pd.read_csv(metadata_path + '/colors.csv', error_bad_lines=False)
    # convert dataframe to dataframe with columns filename, key, value
    colors = pd.melt(colors, id_vars=['filename'], value_name='value')
    # decode base64 string to color string
    colors['value'] = colors['value'].apply(lambda x: base64.b64decode(x).decode('utf-8'))

    # save colors to database (mariadb)
    conn = mysql.connector.connect(**cnf)
    # disable auto commit
    conn.autocommit = False
    cursor = conn.cursor()
    try:
        # create table if it doesn't exist
        cursor.execute('CREATE TABLE IF NOT EXISTS metadata (filename TEXT, mkey TEXT, mvalue TEXT)')
        # insert new colors
        for index, row in colors.iterrows():
            try:
                # check if color already exists (mkey is always 'colors')
                cursor.execute('SELECT * FROM metadata WHERE filename = %s AND mkey = %s AND mvalue = %s',
                               (row['filename'], 'dominant_color', row['value']))
                rows = cursor.fetchall()
                if not rows:
                    # insert color if it doesn't exist
                    cursor.execute('INSERT INTO metadata VALUES (%s, %s, %s)',
                                   (row['filename'], 'dominant_color', row['value']))

            except Error as e:
                continue
        conn.commit()
    except Error as e:
        # print the request error
        return jsonify({'message': 'Error while saving colors to database: {}'.format(e)})
    finally:
        cursor.close()
        conn.autocommit = True
        conn.close()

    # return success message as response
    return jsonify({'message': 'Colors saved successfully'})


if __name__ == '__main__':
    app.run(debug=False)
