import json
import pandas as pd
from dotenv import load_dotenv
import psycopg2.extensions
from pymilvus import DataType, Collection, CollectionSchema, FieldSchema, connections, utility
import ast
import numpy as np
import logging
import spacy

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="postgres",
    port="5432",
    dbname="raw_metadata",
    user="postgres",
    password="postgres"
)


def release_collection(collection_name):
    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        collection.release()
        logging.info("Collection released")


def connect_to_milvus():
    connections.connect(host="standalone", port="19530")

    collection_name = "metadata_vectors"
    if not utility.has_collection(collection_name):
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length_per_row=200, max_length=200),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1516),
            ],
            description="Collection for metadata vectors",
        )
        Collection(name=collection_name, schema=schema)

    # release the collection if is load in memory
    release_collection(collection_name)

    logging.info("Connected to Milvus")
    return collection_name


def create_index(collection_name):
    index_params = {
        "index_type": "IVF_SQ8",
        "metric_type": "L2",
        "params": {"nlist": 1024},
    }

    release_collection(collection_name)
    collection = Collection(name=collection_name)
    collection.create_index(field_name="vector", index_params=index_params)
    logging.info("Index created")


def clean(json_metadata):
    data = pd.DataFrame([json_metadata])

    # dominant color look like that : [["#101b18", 0.480875], ["#1c4536", 0.33805], ["#61aa8a", 0.0403], ["#2e765c", 0.140775]]
    # we want to keep only the first element of the array
    colors = data["dominant_color"].values
    # the colors is a string so we need to convert it to an array
    colors = ast.literal_eval(colors[0])
    # we want to keep only the first element of the array
    updated_colors = []
    for color in colors:
        updated_colors.append(color[0])
    data["dominant_color"] = str(updated_colors)
    data['imagewidth'] = pd.to_numeric(data['imagewidth'], errors='coerce')
    data['imageheight'] = pd.to_numeric(data['imagewidth'], errors='coerce')
    data['orientation'] = pd.to_numeric(data['orientation'], errors='coerce')
    data['imagewidth'].fillna(0, inplace=True)
    data['imageheight'].fillna(0, inplace=True)
    data['orientation'].fillna(0, inplace=True)
    data['make'].fillna("unknown", inplace=True)

    whitelist = ["dominant_color", "imagewidth", "imageheight", "orientation", "tags", "make", "filename"]

    data = data[whitelist]

    return data


def normalize_scale(data, key, scale):
    return data[key].values[0] / scale


def encode_make(make):
    return int.from_bytes(make.encode(), 'little') % 10 ** 10


def extract_rgb(dominant_colors):
    return [color / 255 for sublist in dominant_colors for color in sublist]


def words_to_embeddings(words):
    words = [word.replace(' ', '') for word in words]
    nlp = spacy.load("en_core_web_md")
    # Generate embeddings for each word
    embeddings = []
    for word in words:
        try:
            m = nlp(word).vector
            embeddings.append(m)
        except KeyError:
            embeddings.append(np.zeros(300))
    return embeddings


def preprocess_with_tags(data, max_tags=5):
    width = normalize_scale(data, 'imagewidth', 1000000)
    height = normalize_scale(data, 'imageheight', 1000000)
    orientation = float(data['orientation'].values[0])

    make = data['make'].apply(encode_make)
    make = float(make.values[0])

    dominant_colors = data['dominant_color'].apply(
        lambda x: [int(color[i:i + 2], 16) for color in ast.literal_eval(x) for i in (1, 3, 5)]).tolist()

    rgb_values = extract_rgb(dominant_colors)

    tags = eval(data['tags'].values[0])

    # If the tags list is not equal to 5, fill it with 'N/A' tags
    for i in range(max_tags):
        if len(tags) < max_tags:
            tags.append('N/A')

    embedding = words_to_embeddings(tags)
    # Embedding is a list of list, we want to flatten it
    embedding = [item for sublist in embedding for item in sublist]

    vector = [width, height, orientation, make] + rgb_values + embedding

    return vector


def process_new_metadata(new_metadata):
    # Clean and preprocess new metadata to convert it to a vector
    cleaned_metadata = clean(new_metadata)
    vector = preprocess_with_tags(cleaned_metadata, max_tags=5)
    filename = new_metadata["filename"]
    collection_name = connect_to_milvus()
    collection = Collection(name=collection_name)
    data = [
        [filename],
        [vector],
    ]

    collection.insert(data)

    # flush
    collection.flush()

    # Create an index for the 'vector' field
    create_index(collection_name)

    logging.info("New metadata processed")


logging.info("Listening for new metadata")
# Poll for new notifications and process new metadata

# Set connection to asynchronous mode
conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

# Create a cursor and listen for notifications on the 'metadata_channel'
cur = conn.cursor()
cur.execute("LISTEN metadata_channel;")

while True:
    conn.poll()
    while conn.notifies:
        notify = conn.notifies.pop(0)
        payload = json.loads(notify.payload)
        new_metadata = payload["data"]

        logging.info("New metadata received")
        try:

            process_new_metadata(new_metadata)
        except Exception as e:
            logging.error("Error while processing new metadata: %s", e)
