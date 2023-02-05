from tqdm import tqdm
from PIL import Image
import pickle
import os
import zipfile
import requests
import functools
import pathlib
from tqdm.auto import tqdm

def create_folder(path):
    """
    This function creates a folder at the specified path.
    If the folder already exists, it will print a message saying so.
    If there is an error creating the folder, it will print the error message.

    Parameters:
    path (str): The path of the folder to be created.

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

def download(url, filename):
    try:
        s = requests.Session()
        s.mount(url, requests.adapters.HTTPAdapter(max_retries=3))
        r = s.get(url, stream=True, allow_redirects=True)
        r.raise_for_status()
        file_size = int(r.headers.get('Content-Length', 0))

        path = pathlib.Path(filename).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        desc = "(Unknown total file size)" if file_size == 0 else ""
        r.raw.read = functools.partial(r.raw.read, decode_content=True)
        with tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=desc) as pbar:
            with path.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return path
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred while downloading dataset: {e}")
    except Exception as e:
        print(f"Error occurred while downloading dataset: {e}")

# Need to work correctly : output_folder, image_path, metadata_path
def init_folder(folder_names: list):
    for folder_name in folder_names:
        create_folder(folder_name)

def download_dataset(dataset_url, image_path):
    if not os.path.exists('archive.zip'):
        download(dataset_url, 'archive.zip')
        print("Dataset downloaded!")
        try:
            with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
                zip_ref.extractall(image_path)
                print("Dataset unzipped")
        except Exception as e:
            print(f"Error occurred while unzipping dataset: {e}")
        try:
            os.remove('archive.zip')
            print("archive.zip removed")
        except Exception as e:
            print(f"Error occurred while removing archive.zip: {e}")

def get_all_images(image_path):
    try:
        return [os.path.join(root, name)
                for root, dirs, files in os.walk(image_path)
                for name in files
                if name.endswith((".png", ".jpg"))]
    except Exception as e:
        print(f"An error occurred while fetching images: {e}")
        return []

def create_checkpoint(latest_file):
    try:
        with open('checkpoint.txt', 'w') as f:
            f.write(latest_file)
        print("Checkpoint created successfully")
    except Exception as e:
        print(f"An error occurred while creating checkpoint: {e}")

def load_checkpoint():
    try:
        # first verify if checkpoint exist
        if os.path.exists('checkpoint.txt'):
            with open('checkpoint.txt', 'r') as f:
                return f.read()
        else:
            print("Checkpoint not found")
            return None
    except Exception as e:
        print(f"An error occurred while loading checkpoint: {e}")
        return None

def remove_checkpoint():
    try:
        if os.path.exists('checkpoint.txt'):
            os.remove('checkpoint.txt')
            print("Checkpoint removed successfully")
        else:
            print("Checkpoint not found")
    except Exception as e:
        print(f"An error occurred while removing checkpoint: {e}")

def set_test_dataset(image_path, amount=100):
    try:
        for root, dirs, files in os.walk(image_path, topdown=False):
            for name in files:
                if len(os.listdir(image_path)) > amount:
                    os.remove(os.path.join(root, name))
        print("All images removed except " + str(amount) + " images")
    except Exception as e:
        print(f"An error occurred while setting test dataset: {e}")

def arrange_dataset(image_path, is_test=False):
    try:
        img_files = get_all_images(image_path)
        checkpoint = load_checkpoint()
        for file in tqdm.tqdm(img_files, desc="Moving all file to images folder"):
            if checkpoint == file:
                checkpoint = None
                continue
            elif checkpoint is not None:
                continue
            else:
                os.rename(file, os.path.join(image_path, os.path.basename(file)))
                create_checkpoint(file)
        print("All files moved to images folder")
        remove_checkpoint()

        for root, dirs, files in os.walk(image_path, topdown=False):
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        print("All subfolders removed")

        if is_test:
            set_test_dataset(image_path)
    except Exception as e:
        print("An error occurred while arranging the dataset: ", e)

def get_metadata(img_file, image_path):
    try:
        img_name = os.path.basename(img_file)
        image = Image.open(os.path.join(image_path, img_name))
        exif = image.getexif()

        metadata = {
            "filename": image.filename,
            "size": image.size,
            "height": image.height,
            "width": image.width,
            "format": image.format,
            "mode": image.mode,
        }
    except Exception as e:
        print(f"An error occurred while processing {img_file}: {str(e)}")
        return None

    return metadata

def save_metadata(metadata, img_name, metadata_path):
    try:
        with open(os.path.join(metadata_path, os.path.splitext(os.path.basename(img_name))[0] + '.pickle'), 'wb') as f:
            pickle.dump(metadata, f)
    except Exception as e:
        print(f"An error occurred while saving metadata for {img_name}: {str(e)}")

def get_all_metadata(image_path, metadata_path):
    img_files = get_all_images(image_path)
    # Get all metadata of the images and save to individual json file
    checkpoint = load_checkpoint()
    for img in tqdm(img_files, desc="Get all metadata of the images and save to individual pikkle file"):
        if checkpoint == img:
            checkpoint = None
            continue
        elif checkpoint is not None:
            continue
        else:
            metadata = get_metadata(img, image_path)
            if metadata:
                save_metadata(metadata, img, metadata_path)
                create_checkpoint(img)
    remove_checkpoint()

