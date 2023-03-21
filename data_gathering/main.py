import os
import zipfile
import requests
import functools
import pathlib
from tqdm import tqdm
import sqlite3
import pandas as pd
import nest_asyncio
import aiohttp
import time
import asyncio

# Set the base folder path for the project
output_path = "../output"
images_path = os.path.join(output_path, "images")
metadata_path = os.path.join(output_path, "metadata")
config_path = os.path.join(output_path, "config")

list_of_paths = [output_path, images_path, metadata_path, config_path]

# Set the base URL for the dataset
dataset_url = "w"
# metadata mode (used to save metadata)
metadata_mode = "sqlite"

nest_asyncio.apply()

def create_folder(path):
    """
    This function creates a folder at the specified path.
    If the folder already exists, it will print a message saying so.
    If there is an error creating the folder, it will print the error message.

    Parameters:
        :param path (str): The path of the folder to be created.

    Returns:
    None
    """
    try:
        # Use os.mkdir to create the folder at the specified path
        os.mkdir(path)
        print(f"Folder {path} created")
    except FileExistsError:
        # If the folder already exists, print a message saying so
        print(f"Folder {path} already exists")
    except Exception as e:
        # If there is an error creating the folder, print the error message
        print(f"Error creating folder {path}: {e}")

def init_folder(folder_names: list):
    for folder_name in folder_names:
        create_folder(folder_name)


def download(url, filename):
    """
    This download a file from a given URL and save it to a specified filename.

    Parameters:
        :param url (str): The URL of the file to be downloaded.
        :param filename (str): The filename to save the file as.

    Returns:
    path (str): The path of the downloaded file.
    """
    try:
        # Create a session object to persist the state of connection
        s = requests.Session()
        # Retry connecting to the URL up to 3 times
        s.mount(url, requests.adapters.HTTPAdapter(max_retries=3))
        # Send a GET request to the URL to start the download
        r = s.get(url, stream=True, allow_redirects=True)
        # Raise an error if the response is not 200 OK
        r.raise_for_status()
        # Get the file size from the Content-Length header, default to 0 if not present
        file_size = int(r.headers.get('Content-Length', 0))
        # Get the absolute path to the target file
        path = pathlib.Path(filename).expanduser().resolve()
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        # Set the description to display while downloading, "(Unknown total file size)" if file size is 0
        desc = "(Unknown total file size)" if file_size == 0 else ""
        # Enable decoding the response content
        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        # Use tqdm to display the download progress
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc) as pbar:
            # Open the target file in binary write mode
            with path.open("wb") as f:
                # Write each chunk of data from the response to the file
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
        # Return the path to the downloaded file
        return path
    # Handle HTTP error if the response is not 200 OK
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while downloading dataset: {e}")
    # Handle any other exceptions that might occur while downloading the file
    except Exception as e:
        print(f"Error occurred while downloading dataset: {e}")

def download_dataset(dataset_url, image_path):
    """
    Downloads the dataset from the given URL, unzips it, and stores the images in the specified image path.

    Args:
        :param dataset_url (str): URL of the dataset to be downloaded
        :param image_path (str): Path to store the images after unzipping the dataset
    """
    # Check if the dataset has already been downloaded
    # Check if the archive.zip file exists or if the images folder is empty
    if not os.path.exists('archive.zip'):
        # Download the dataset from the given url
        download(dataset_url, 'archive.zip')
        print("Dataset downloaded!")
        try:
            # Extract the contents of the archive.zip to the specified image path
            with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
                zip_ref.extractall(image_path)
            print("Dataset unzipped")
        except Exception as e:
            print(f"Error occurred while unzipping dataset: {e}")
        try:
            # Remove the archive.zip file
            os.remove('archive.zip')
            print("archive.zip removed")
        except Exception as e:
            print(f"Error occurred while removing archive.zip: {e}")


async def download_image(session: aiohttp.ClientSession, url: str, i: int, err_cnt=None):
    """
    Downloads an image from the given URL using an aiohttp client session and saves it to the local file system.

    Args:
        session: An aiohttp client session that manages HTTP requests and responses.
        url: The URL of the image to download.
        i: An integer representing the index of the image to download.
        err_cnt: An optional integer representing the number of times that the download has failed due to a client error.
                 If not provided, it defaults to 0.

    Raises:
        This method does not raise any exceptions.

    Returns:
        None.
    """
    if err_cnt is None:
        err_cnt = 0
    try:
        async with session.get(url) as response:
            filename = os.path.join(images_path, "image_" + str(i) + ".jpg")
            with open(filename, 'wb') as f:
                f.write(await response.content.read())
            print(f"Downloaded {url} to {filename} idx: {i}")
    except aiohttp.ClientError as e:
        print(f"Error occurred while downloading {url}: {e}")
        if err_cnt == 10:
            return
        await asyncio.sleep(10)
        err_cnt += 1
        await download_image(session, url, i, err_cnt)


async def download_images(image_urls, images_ids):
    """
    Downloads a list of images from the given URLs using an aiohttp client session and saves them to the local file system.

    Args:
        image_urls: A list of strings representing the URLs of the images to download.
        images_ids: A list of integers representing the indices of the images to download.

    Raises:
        This method does not raise any exceptions.

    Returns:
        None.
    """
    # Create a new aiohttp client session to manage HTTP requests and responses
    async with aiohttp.ClientSession() as session:
        tasks = []  # Create an empty list to hold the tasks that will download the images
        semaphore = asyncio.Semaphore(5000)  # Create a semaphore to limit the number of concurrent downloads
        # Loop through the image URLs and create a new task for each one
        for i, url in enumerate(image_urls):
            try:
                await semaphore.acquire()  # Acquire a permit from the semaphore to limit concurrency
                #url = url + "?w=1000&fm=jpg&fit=max"  # Append query parameters to resize and optimize the image
                task = asyncio.ensure_future(download_image(session, url, images_ids[i]))  # Create a new download task
                task.add_done_callback(
                    lambda x: semaphore.release())  # Release the semaphore permit when the task completes
                tasks.append(task)  # Add the task to the list of download tasks
            except Exception:
                print(f"Error occurred while downloading {url}")
                semaphore.release()  # Release the semaphore permit if an exception occurs
        # Wait for all download tasks to complete
        await asyncio.wait(tasks)
        # Gather the results of all download tasks (not necessary because the tasks have already completed)
        await asyncio.gather(*tasks)


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


# Method to create a checkpoint with the latest file
def create_checkpoint(latest_file):
    """
   Creates a checkpoint file containing the latest processed file name.

   Parameters:
       :param latest_file (str): The name of the latest processed file.

   Returns:
       None
   """
    try:
        # Open a file in write mode
        with open('checkpoint.txt', 'w') as f:
            # Write the latest file to the checkpoint
            f.write(latest_file)
    except Exception as e:
        # Print error message
        print(f"An error occurred while creating checkpoint: {e}")


# Method to load a checkpoint
def load_checkpoint():
    """
    Loads the checkpoint file if it exists.

    Returns:
        str: The name of the latest processed file, None if checkpoint file not found.
    """
    try:
        # Check if checkpoint exists
        if os.path.exists('checkpoint.txt'):
            # Open the checkpoint in read mode
            with open('checkpoint.txt', 'r') as f:
                # Return the contents of the checkpoint
                return f.read()
        else:
            # Print message if checkpoint not found
            print("Checkpoint not found")
            return None
    except Exception as e:
        # Print error message
        print(f"An error occurred while loading checkpoint: {e}")
        return None


# Method to remove a checkpoint
def remove_checkpoint():
    """
    Removes the checkpoint file if it exists.

    Returns:
        None
    """
    try:
        # Check if checkpoint exists
        if os.path.exists('checkpoint.txt'):
            # Remove the checkpoint
            os.remove('checkpoint.txt')
            # Print success message
            print("Checkpoint removed successfully")
        else:
            # Print message if checkpoint not found
            print("Checkpoint not found")
    except Exception as e:
        # Print error message
        print(f"An error occurred while removing checkpoint: {e}")


def set_test_dataset(image_path, amount=100):
    """
    This function removes all images from the given image_path except the first amount images.

    Parameters:
    :param image_path (str): The path to the image folder.
    :param amount (int, optional): The number of images to keep in the folder. Defaults to 100.

    Returns:
    None
    """
    try:

        # loop through the images in the directory using tqdm
        for root, dirs, files in os.walk(image_path, topdown=False):
            for name in tqdm(files, desc="Removing images"):
                # check if the number of images is greater than the amount specified
                if len(os.listdir(image_path)) > amount:
                    # remove the image if the number of images is greater than the specified amount
                    os.remove(os.path.join(root, name))
        # print message indicating that images were removed successfully
        print("All images removed except " + str(amount) + " images")
    except Exception as e:
        # print error message if there was an error during removal of images
        print(f"An error occurred while setting test dataset: {e}")


def arrange_dataset(image_path, is_test=False):
    """
    Arrange the dataset stored in `image_path`.

    :param image_path: path to the dataset folder.
    :param is_test: If True, the dataset will be set to a test set with only 100 images.
    """
    try:
        # Get a list of all images in the path
        img_files = get_all_images(image_path)
        # Load the last checkpoint if it exists
        checkpoint = load_checkpoint()
        # Iterate over all image files
        for file in tqdm(img_files, desc="Moving all file to images folder"):
            # Check if the current file matches the checkpoint
            if checkpoint == file:
                # If it does, reset the checkpoint
                checkpoint = None
                continue
            # If the checkpoint is not None, skip this file
            elif checkpoint is not None:
                continue
            # If neither of the above conditions are met, move the file
            else:
                os.rename(file, os.path.join(image_path, os.path.basename(file)))
                # Create a new checkpoint after moving the file
                create_checkpoint(file)
        # Print a message indicating that all files have been moved
        print("All files moved to images folder")
        # Remove the checkpoint since all files have been moved
        remove_checkpoint()

        # Remove all subfolders in the image path
        for root, dirs, files in os.walk(image_path, topdown=False):
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        # Print a message indicating that all subfolders have been removed
        print("All subfolders removed")

        print(is_test)
        # If is_test is True, set the test dataset
        if is_test:
            print("Setting test dataset 13&3" + image_path)
            set_test_dataset(image_path)
            print("Test dataset set successfully")
    # Catch any exceptions that may occur
    except Exception as e:
        print("An error occurred while arranging the dataset: ", e)


async def get_metadata(img_file):
    """
    This coroutine extracts metadata information from a list of image files and returns it in a dictionary format.

    Parameters:
    img_files (str): A string containing all image file paths separated by a space.

    Returns:
    dict: A dictionary containing the metadata information of the images. If an error occurs for any image, the metadata for that image will be None.
    """
    try:
        import PIL.Image
        # get tags
        from PIL.ExifTags import TAGS

        whitelist = ['Make', 'DateTimeOriginal', 'Width', 'Height', 'File Name', 'Artist', 'GPSInfo', 'Orientation']

        img = PIL.Image.open(img_file)
        exif_data = img._getexif()
        exif_data = {TAGS[k]: v for k, v in exif_data.items() if k in TAGS}
        exif_data['File Name'] = os.path.basename(img_file)
        exif_data['Width'] = img.width
        exif_data['Height'] = img.height

        # clean up the dictionary by removing all keys that are not in the whitelist
        exif_data = {k: v for k, v in exif_data.items() if k in whitelist}
        # return the dictionary
        return exif_data
    except Exception as e:
        # print an error message if an error occurs
        print(f"An error occurred while getting metadata for {img_file}: {e}")
        return None


def gen_sql_requests(metadatas):
    """
    This function generates a list of SQL requests to insert metadata into a database.

    Parameters:
    metadatas (list): A list of dictionaries containing the metadata information of the images.

    Returns:
    list: A list of SQL requests to insert metadata into a database.
    """
    # Create a list to store SQL requests
    sql_requests = []

    # Loop over all metadata
    for metadata in tqdm(metadatas, desc="Generating SQL requests"):
        try:
            # Get the filename of the image
            filename = metadata['File Name']

            # Loop over all metadata items
            for key, value in metadata.items():
                # Create SQL request to insert metadata into database
                # replace " by space
                value = value.replace('"', ' ')

                sql_request = f"INSERT INTO metadata VALUES ('{filename}', '{key}', '{value}')"
                # Add SQL request to list
                sql_requests.append(sql_request)

        except Exception as e:
            # Print an error message if an error occurs
            print("An error occurred while generating SQL requests: ", e)
            continue
    # Return the list of SQL requests
    return sql_requests

# Execute sql query
def execute_query(queries_array):
    conn = sqlite3.connect(os.path.join(metadata_path, 'metadata.db'))
    # Set the cache size to 100000 pages
    conn.execute("PRAGMA cache_size = 100000")
    # Set the synchronous mode to OFF
    conn.execute("PRAGMA synchronous = OFF")
    # Set the journal mode to WAL
    conn.execute("PRAGMA journal_mode = WAL")
    # check if the table metadata exists in the database else create it
    if conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone() is None:
        conn.execute("CREATE TABLE metadata (filename text, key text, value text)")
        conn.commit()

    # Insert the metadata into the database
    conn.executescript(';'.join(queries_array))
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()


async def get_all_metadata(image_path):
    """
    This coroutine extracts metadata from all images in a directory and saves the metadata information in either pickle or json format.

    Parameters:
    image_path (str): The path to the directory where the images are stored.
    metadata_path (str): The path to the directory where the metadata will be saved.

    Returns:
    None
    """
    # Get a list of all images in the directory
    img_files = get_all_images(image_path)

    metadata_list = []
    for img_file in tqdm(img_files, desc="Extracting metadata"):
        try:
            metadata = await get_metadata(img_file)
            if metadata is not None:
                metadata_list.append(metadata)
        except Exception as e:
            print("Error extracting metadata: ", e, " for file: ", img_file)
            continue

    # filter none values
    metadata_list = list(filter(None, metadata_list))

    queries = gen_sql_requests(metadata_list)

    # save queries to requests.txt
    with open(os.path.join(metadata_path, 'requests.txt'), 'w') as f:
        for query in queries:
            f.write('\n'.join(query))


    execute_query(queries)


asyncio.run(get_all_metadata(images_path))