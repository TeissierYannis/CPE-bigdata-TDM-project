import json
import pandas as pd
from dotenv import load_dotenv
import psycopg2.extensions
from pymilvus import DataType, Collection, CollectionSchema, FieldSchema, connections, utility, IndexType
import ast
import numpy as np
import logging
from transformers import PreTrainedTokenizerFast

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

labels = ['N/A', 'person', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench',
          'bird', 'cat', 'dog', 'horse', 'bicycle', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat',
          'backpack', 'umbrella', 'shoe', 'car', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
          'snowboard', 'sports ball', 'kite', 'baseball bat', 'motorcycle', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'airplane', 'spoon', 'bowl',
          'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'bus', 'donut', 'cake',
          'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'train', 'toilet',
          'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'truck', 'toaster',
          'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'boat',
          'toothbrush']

tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")
tokenizer.add_tokens(labels)


def custom_padding(tokens, max_tags):
    if len(tokens) < max_tags:
        padding = [0] * (max_tags - len(tokens))
        tokens.extend(padding)
    else:
        tokens = tokens[:max_tags]
    return tokens


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
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=21),
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


def tokenize_tags(tokenizer, tags, max_tags):
    if not tags:
        tags = ['N/A'] * max_tags
    tokens = tokenizer.convert_tokens_to_ids(tags)
    tokens = custom_padding(tokens, max_tags)

    # replace None by 0
    tokens = [0 if x is None else x for x in tokens]

    return tokens


def preprocess_with_tags(data, tokenizer, max_tags=5):
    width = normalize_scale(data, 'imagewidth', 1000000)
    height = normalize_scale(data, 'imageheight', 1000000)
    orientation = float(data['orientation'].values[0])

    make = data['make'].apply(encode_make)
    make = float(make.values[0])

    dominant_colors = data['dominant_color'].apply(
        lambda x: [int(color[i:i + 2], 16) for color in ast.literal_eval(x) for i in (1, 3, 5)]).tolist()

    rgb_values = extract_rgb(dominant_colors)

    tags = eval(data['tags'].values[0])
    tokenized_tags = tokenize_tags(tokenizer, tags, max_tags)
    vector = [width, height, orientation, make] + rgb_values + tokenized_tags

    return vector


def process_new_metadata(new_metadata):
    # Clean and preprocess new metadata to convert it to a vector
    cleaned_metadata = clean(new_metadata)
    vector = preprocess_with_tags(cleaned_metadata, tokenizer, max_tags=5)
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


# Set connection to asynchronous mode
conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

# Create a cursor and listen for notifications on the 'metadata_channel'
cur = conn.cursor()
cur.execute("LISTEN metadata_channel;")

logging.info("Listening for new metadata")
# Poll for new notifications and process new metadata
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
