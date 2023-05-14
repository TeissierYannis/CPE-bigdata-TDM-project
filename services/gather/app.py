import asyncio
import io
import uuid
from typing import List

import aiohttp
import os
import pandas as pd
from flask import Flask, jsonify, request
import threading
from PIL import Image
from minio import Minio, S3Error

from .classes import sharedprogress

app = Flask(__name__)

# Initialize the MinIO client
minio_client = Minio(
    "minio:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)

# Path to save images
bucket_name = 'images'

# Ensure the Minio bucket exists
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)


async def send_to_harvest(session: aiohttp.ClientSession, image_filename: str) -> None:
    # TODO, replace with env
    async with session.post('http://gateway/harvest/process',
                            json={"filename": image_filename}) as harvest_response:
        harvest_response_data = await harvest_response.text()
        print(f"Harvest response for {image_filename}: {harvest_response_data}")


async def download_images(session: aiohttp.ClientSession, urls: List[str]) -> None:
    async def download_image(session: aiohttp.ClientSession, url: str) -> None:
        async with session.get(url) as response:
            image_data = await response.read()
            image_filename = url.split('/')[-1] + '.jpg'

            # Save image to Minio
            minio_client.put_object(
                bucket_name,
                image_filename,
                io.BytesIO(image_data),
                len(image_data),
                content_type='image/jpeg'
            )

            # Send an HTTP request to http://127.0.0.1:81/harvest/
            asyncio.create_task(send_to_harvest(session, image_filename))

    semaphore = asyncio.Semaphore(1)
    tasks = []

    def release_and_update():
        semaphore.release()
        sharedprogress.SharedProgress().set_progress(sharedprogress.SharedProgress().get_progress() + 1)

    for url in urls:
        try:
            if sharedprogress.SharedProgress().get_status() == 'cancelled':
                break
            await semaphore.acquire()
            task = asyncio.ensure_future(download_image(session, url))
            task.add_done_callback(lambda x: release_and_update())
            tasks.append(task)
        except Exception:
            print(f"Error occurred while downloading {url}")
            semaphore.release()

    await asyncio.wait(tasks)
    sharedprogress.SharedProgress().set_status('completed')


def start_download_task(urls):
    sharedprogress.SharedProgress().reset()
    sharedprogress.SharedProgress().set_status('in progress')
    sharedprogress.SharedProgress().set_total(len(urls))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(download_images(aiohttp.ClientSession(), urls))


@app.route('/download')
async def download():
    if sharedprogress.SharedProgress().get_status() == 'in progress':
        return {'status': 'error', 'message': 'Download task is already running'}

    try:
        full_path = os.path.join(os.getcwd(), 'config', 'photos.tsv000')
        photo_df = pd.read_csv(full_path, sep='\t')
    except Exception as e:
        request.status_code = 500
        return jsonify(
            {'status': 'error', 'message': 'Urls file not found'}
        )

    photo_df = photo_df[photo_df['photo_image_url'].notnull()]
    threading.Thread(target=start_download_task, args=(photo_df['photo_image_url'].tolist(),)).start()

    return {'status': 'success', 'message': 'Download started'}


@app.route('/status')
def status():
    """
     This method will allow users to check the status of the download. It could return information such as the current status of the download (e.g., in progress, complete, failed), the percentage of the download that has been completed, and any error messages that have occurred.
    :return:
    """
    # Check if download task is running
    print(sharedprogress.SharedProgress().get_status())
    if sharedprogress.SharedProgress().get_status() == 'in progress':
        return {'status': 'in progress',
                'message': f"{sharedprogress.SharedProgress().get_progress()} of {sharedprogress.SharedProgress().get_total()} images downloaded"}

    if sharedprogress.SharedProgress().get_status() == 'completed':
        return {'status': 'completed', 'message': 'Download completed'}

    if sharedprogress.SharedProgress().get_status() == 'cancelled':
        return {'status': 'error', 'message': f"Error occurred while downloading images"}

    return {'status': 'error', 'message': 'No download task is currently running'}


def send_to_harvest_async(filename):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(send_to_harvest(aiohttp.ClientSession(), filename))
    finally:
        loop.close()


@app.route('/uploads', methods=['POST'])
def uploads():
    # TODO: Try to handle more file (in quantity, after more than 10mb it fails and the container crashes)
    """
    This method will allow users to upload multiple images to the service. The images will be stored in the MinIO bucket.
    :return:
    """
    if 'files[]' not in request.files:
        return jsonify(status='error', message='No file part')

    files = request.files.getlist('files[]')

    # Open files and upload to Minio with Pillow
    PIL_images = []
    for file in files:
        # is jpeg or jpg
        if file and file.filename.split('.')[-1].lower() in ['jpeg', 'jpg']:
            PIL_images.append(Image.open(file))

    if len(files) == 0:
        return jsonify(status='error', message='No selected file')

    # Save images to Minio
    for PIL_image in PIL_images:
        with io.BytesIO() as output:
            PIL_image.save(output, format="JPEG")
            output.seek(0)  # Reset the output's file pointer to the beginning
            minio_client.put_object(
                bucket_name,
                str(uuid.uuid4()) + '.jpg',
                output,
                len(output.getvalue()),
                content_type='image/jpeg'
            )

    # Send an HTTP request to http://
    for PIL_image in PIL_images:
        filename = str(uuid.uuid4()) + '.jpg'
        PIL_image.save(filename)
        threading.Thread(target=send_to_harvest_async, args=(filename,)).start()

    return jsonify(status='success', message='Files uploaded')


@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    sharedprogress.SharedProgress().set_status('stopped')
    app.run(debug=False)
