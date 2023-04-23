import os
import sys

import pandas as pd
from flask import Flask, jsonify, request
from mysql.connector import pooling
from dotenv import load_dotenv
from collections import namedtuple
import ast
from pymilvus import DataType, Collection, CollectionSchema, FieldSchema, connections, utility, IndexType, SearchResult
import numpy as np
import logging
from transformers import PreTrainedTokenizerFast
from sklearn.preprocessing import OneHotEncoder

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

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


def custom_padding(tokens, max_tags):
    if len(tokens) < max_tags:
        padding = [0] * (max_tags - len(tokens))
        tokens.extend(padding)
    else:
        tokens = tokens[:max_tags]
    return tokens


def normalize_scale(value, scale):
    return value / scale


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


def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def vectorize_preferences(preferences):
    width = normalize_scale(preferences['imagewidth'], 1000000)
    height = normalize_scale(preferences['imageheight'], 1000000)
    orientation = float(1) if preferences['orientation'] == 'landscape' else float(0)
    make = encode_make(preferences['make'])

    rgb_color = hex_to_rgb(preferences['dominant_color'])

    logging.info("RGB: {}".format(rgb_color))

    r, g, b = rgb_color
    r = normalize_scale(r, 255)
    g = normalize_scale(g, 255)
    b = normalize_scale(b, 255)

    tags = preferences['tags']
    # Remove duplicates
    tags = list(dict.fromkeys(tags))
    logging.info("Tags: {}".format(tags))

    # Use OneHotEncoder to encode the tags (but use only the first 5)
    max_tags = 5
    labels_array = np.array(labels).reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels_array)
    tags_array = np.array(tags).reshape(-1, 1)
    result = encoder.transform(tags_array)
    logging.info("Tags: {}".format(result))
    # convert to array of float
    tokenized_tags = result.toarray().tolist()[0]
    logging.info("Tags: {}".format(tokenized_tags))
    tokenized_tags = custom_padding(tokenized_tags, max_tags)

    logging.info("Width: {}".format(width))
    logging.info("Height: {}".format(height))
    logging.info("Orientation: {}".format(orientation))
    logging.info("Make: {}".format(make))
    logging.info("R: {}".format(r))
    logging.info("G: {}".format(g))
    logging.info("B: {}".format(b))
    logging.info("Tags: {}".format(tokenized_tags))

    vector = [width, height, orientation, make, r, g, b, r, g, b, r, g, b, r, g, b] + tokenized_tags

    return vector


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get the preferences from the request
        data = request.get_json()
        # {'dominant_color': '#000000', 'imagewidth': 800, 'imageheight': 600, 'orientation': 'landscape', 'tags': ['nature', 'landscape'], 'make': 'Canon'}

        preferences = data["preferences"]

        # Convert the preferences to a vector
        vector = vectorize_preferences(preferences)

        logging.info("Vector: {}".format(vector))

        # Connect to Milvus
        collection_name = connect_to_milvus()

        # Load the metadata vectors
        collection = Collection(name=collection_name)
        collection.load()

        # Get the top 10 recommendations
        query_vector = [vector]
        search_param = {
            "data": query_vector,
            "anns_field": "vector",
            "param": {"metric_type": "L2", "params": {"nprobe": 1024}},
            "limit": 10,
        }

        results = collection.search(**search_param)

        logging.info("Results: {} : {}".format(type(results), results))

        # Get the primary keys (IDs) of the top 10 nearest neighbors
        top_ids = [result.id for result in results[0]]

        logging.info("Top IDs: {} : {}".format(type(top_ids), top_ids))

        # Fetch filenames corresponding to the primary keys (IDs) using query()
        id_list_str = ", ".join(map(str, top_ids))
        expr = f"id in [{id_list_str}]"
        output_fields = ["id", "filename"]
        query_results = collection.query(expr, output_fields=output_fields)

        logging.info("Query Results: {} : {}".format(type(query_results), query_results))

        # Create a mapping of primary keys (IDs) to filenames
        id_to_filename = {result["id"]: result["filename"] for result in query_results}

        logging.info("ID to Filename: {} : {}".format(type(id_to_filename), id_to_filename))

        # release
        collection.release()

    except:
        # Log details of the exception and request data
        logging.error("Request : {} {}".format(request.data, request.args))
        logging.error(request.values)
        logging.error("Unexpected error: {}".format(sys.exc_info()[0]))
        return jsonify({"error": "Unexpected error: {}".format(sys.exc_info()[0])})

    return jsonify(id_to_filename)


if __name__ == '__main__':
    app.run(debug=False)
