import asyncio
import imghdr
import io

import aiohttp
import os
import pandas as pd
from typing import List
from flask import Flask, jsonify, request, Response
import threading
from .classes import sharedprogress
from PIL import Image

app = Flask(__name__)

# Path to save images
images_path = '/app/shared/images'


async def download_images(session: aiohttp.ClientSession, urls: List[str]) -> None:
    """
    This function downloads all images from the given list of urls.

    Args:
    - session (aiohttp.ClientSession): session object to use for the requests.
    - urls (List[str]): list of URLs of the images to download.
    - download_status (dict): dictionary to store download status.

    Returns:
    None
    """
    async def download_image(session: aiohttp.ClientSession, url: str) -> bytes:
        """
        This function downloads an image from the given url using the given session object.

        Args:
        - session (aiohttp.ClientSession): session object to use for the request.
        - url (str): URL of the image to download.

        Returns:
        - bytes: content of the downloaded image.
        """
        async with session.get(url) as response:
            # write the image to a file, while the image is not fully downloaded set extension to .part and rename it to
            # '.jpg' when download is complete
            with open(os.path.join(images_path, url.split('/')[-1] + '.part'), 'wb') as f:
                while True:
                    chunk = await response.content.read(1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.rename(os.path.join(images_path, url.split('/')[-1] + '.part'),
                      os.path.join(images_path, url.split('/')[-1] + '.jpg'))
            return await response.read()

    semaphore = asyncio.Semaphore(1)  # Limit number of concurrent downloads
    tasks = []

    def release_and_update():
        """
        This function releases the semaphore permit and updates the download status dictionary.
        :return: None
        """
        semaphore.release()
        # Update download status queue
        sharedprogress.SharedProgress().set_progress(sharedprogress.SharedProgress().get_progress() + 1)

    for url in urls:
        try:
            if sharedprogress.SharedProgress().get_status() == 'cancelled':
                break
            await semaphore.acquire()
            task = asyncio.ensure_future(download_image(session, url))  # Create a new download task
            task.add_done_callback(
                lambda x: release_and_update())  # Release the semaphore permit when the task completes
            tasks.append(task)  # Add the task to the list of download tasks
        except Exception:
            print(f"Error occurred while downloading {url}")
            semaphore.release()  # Release the semaphore permit if an exception occurs

    # Wait for all download tasks to complete
    await asyncio.wait(tasks)

    sharedprogress.SharedProgress().set_status('completed')


def start_download_task(urls):
    """
    This function starts the download task in a background thread.
    :param urls: list of URLs of the images to download.
    :return: None
    """
    # init queue to store download status
    sharedprogress.SharedProgress().reset()
    # set status to in progress
    sharedprogress.SharedProgress().set_status('in progress')
    # set total number of images to download
    sharedprogress.SharedProgress().set_total(len(urls))
    # Download images
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(download_images(aiohttp.ClientSession(), urls))

    # Remove partially downloaded files
    partial_files = [f for f in os.listdir(images_path) if f.endswith('.part')]
    for partial_file in partial_files:
        os.remove(os.path.join(images_path, partial_file))


@app.route('/download')
async def download():
    """
    This method will handle the downloading of the dataset of images. It reads the file photos.tsv0000 and downloads all images in the photo_image_url column.
    :return:
    """
    # Create images directory if it doesn't exist
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Check if download task is already running
    if sharedprogress.SharedProgress().get_status() == 'in progress':
        return {'status': 'error', 'message': 'Download task is already running'}

    # Read photos.tsv0000 file
    try:
        # full path to the file
        full_path = os.path.join(os.getcwd(), 'config', 'photos.tsv000')
        photo_df = pd.read_csv(full_path, sep='\t')

    except Exception as e:
        # set status to error and return
        request.status_code = 500
        return jsonify(
            {'status': 'error', 'message': 'Urls file not found'}
        )

    # Filter for images with valid URLs
    photo_df = photo_df[photo_df['photo_image_url'].notnull()]

    # Create new download task in a background thread
    threading.Thread(target=start_download_task, args=(photo_df['photo_image_url'].tolist()[:100],)).start()

    # return status
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


@app.route('/cancel')
def cancel():
    """
    This method will allow users to cancel the download if it is still in progress. It could accept parameters such as the ID of the download, and return a message indicating whether the download was successfully cancelled.
    :return:
    """
    # Stop the download task if it is running
    try:
        # TODO Check if download task is running
        if sharedprogress.SharedProgress().get_status() == 'in progress':
            sharedprogress.SharedProgress().set_status('cancelled')
            return {'status': 'success', 'message': 'Download cancelled'}
    except Exception:
        pass

    return {'status': 'error', 'message': 'No download task is currently running'}


@app.route('/list')
def list_downloads():
    """
    This method will allow users to view a list of all downloads that have been requested, along with their current status. It could return information such as the ID of the download, the URL of the dataset, the location where the dataset is being saved, and the current status of the download.

    :return:
    """

    downloads = []

    # Check if images directory exists
    if not os.path.exists(images_path):
        return {'status': 'error', 'message': 'Images directory not found'}

    # Get list of files in the images directory
    file_list = os.listdir(images_path)
    for file in file_list:
        file_path = os.path.join(images_path, file)
        if os.path.isfile(file_path):
            downloads.append(file)

    return {'status': 'success', 'message': 'Download files listed', 'downloads': downloads}


@app.route('/delete')
def delete():
    """
    This method will allow users to delete a download that has already been completed. It could accept parameters such as the ID of the download, and return a message indicating whether the download was successfully deleted.
    :return:
    """
    # Check if images directory exists
    if not os.path.exists(images_path):
        return {'status': 'error', 'message': 'Images directory not found'}

    # Remove all files in the images directory
    file_list = os.listdir(images_path)
    for file in file_list:
        file_path = os.path.join(images_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error occurred while deleting {file_path}: {e}")

    return {'status': 'success', 'message': 'Download files deleted'}


@app.route('/show/<filename>', methods=['GET'])
def show(filename):
    """
    This method will allow users to view the downloaded images.
    :param filename: name of the file to show
    :return: image
    """
    # Check if images directory exists
    if not os.path.exists(images_path):
        return {'status': 'error', 'message': 'Images directory not found'}

    # Check if file exists
    file_path = os.path.join(images_path, filename)
    if not os.path.exists(file_path):
        return {'status': 'error', 'message': 'File not found'}

    # Check if file is an image
    if not imghdr.what(file_path):
        return {'status': 'error', 'message': 'File is not an image'}

    # Save image to buffer and return it
    buffer = io.BytesIO()
    img = Image.open(file_path)
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return Response(buffer.getvalue(), mimetype='image/jpeg')



    # Send image as base64 encoded string
    #with open(file_path, "rb") as image_file:
    #    encoded_string = base64.b64encode(image_file.read())

    #return {'status': 'success', 'message': 'Image shown', 'image': encoded_string.decode('utf-8')}


# Used to download the file from the server
def send_file(file_path):
    """
    This method will send the file to the client
    :param file_path: path of the file to send
    :return: file
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    response = Response(data, mimetype='image/jpeg')
    response.headers['Content-Disposition'] = 'attachment; filename=' + file_path
    return response


if __name__ == '__main__':
    sharedprogress.SharedProgress().set_status('stopped')
    app.run(debug=False)
