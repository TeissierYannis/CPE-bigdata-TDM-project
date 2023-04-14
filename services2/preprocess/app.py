import ast
import json
import pandas as pd
import numpy as np

from dotenv import load_dotenv
import psycopg2.extensions
import logging
from pymilvus import DataType, Collection, CollectionSchema, FieldSchema, connections, utility, IndexType

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


def connect_to_milvus():
    connections.connect(host="standalone", port="19530")

    collection_name = "metadata_vectors"
    if not utility.has_collection(collection_name):
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length_per_row=200, max_length=200),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            ],
            description="Collection for metadata vectors",
        )
        collection = Collection(name=collection_name, schema=schema)

    logging.info("Connected to Milvus")
    return collection_name


def create_index(collection_name):
    index_params = {
        "index_type": "IVF_SQ8",
        "metric_type": "L2",
        "params": {"nlist": 1024},
    }

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
    logging.info("3Dominant color: %s", data["dominant_color"])
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


def preprocess(data):
    # 1. Normalize and scale numerical properties
    # Width and Height: scale to [0, 1] by dividing each value by the maximum value
    # Hex colors: Convert hex colors to RGB values in the range 0-255 and normalize it by dividing by 255 (range [0, 1])

    data['imagewidth'] = data['imagewidth'] / data['imagewidth'].max()  # TODO hum any max here?
    data['imageheight'] = data['imageheight'] / data['imageheight'].max()  # TODO hum any max here?

    # Convert hex color to RGB and normalize
    # Remove # from hex color
    # print dominant color at each step
    logging.info("Dominant color: %s", data["dominant_color"])
    data["dominant_color"] = data["dominant_color"].apply(lambda x: x[0][1:] if len(x) > 0 else None)
    logging.info("Dominant color: %s", data["dominant_color"])
    data["rgb_color"] = data["dominant_color"].apply(
        lambda x: tuple(int(x[i:i + 2], 16) for i in (0, 2, 4)) if x is not None else None)
    logging.info("RGB color: %s", data["rgb_color"])
    data = data.dropna(subset=["rgb_color"])
    data[["r", "g", "b"]] = pd.DataFrame(data["rgb_color"].tolist(), index=data.index) / 255
    logging.info("RGB color: %s", data[["r", "g", "b"]])
    data.drop(["dominant_color", "rgb_color"], axis=1, inplace=True)

    # One-hot encode categorical properties
    # 1. tags, make, orientation
    # eval tags to have an array and get only the first index of subarray
    convert_to_array = lambda tag_string: ast.literal_eval(tag_string)[0]

    data["tags"] = data["tags"].apply(convert_to_array)
    logging.info("Tags: %s", data["tags"])

    data = pd.concat([data, pd.get_dummies(data["tags"].apply(pd.Series).stack()).sum(level=0)], axis=1)
    data = pd.concat([data, pd.get_dummies(data["make"], prefix="make")], axis=1)
    data["landscape"] = data["orientation"].apply(lambda x: 1 if x == 0 else 0)
    data["portrait"] = data["orientation"].apply(lambda x: 1 if x == 1 else 0)
    data.drop(["tags", "make", "orientation", "filename"], axis=1, inplace=True)

    logging.info("Metadata cleaned and preprocessed")

    return np.concatenate(data.values)


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# TODO: Refacto this method
def preprocess_data(df, max_pad_size):
    def pad_vector(vector, target_size):
        if len(vector) > target_size:
            print(f"Input vector length: {len(vector)}, target size: {target_size}")
            raise ValueError("The input vector is larger than the target size.")
        return np.pad(vector, (0, target_size - len(vector)), 'constant', constant_values=0)

    def preprocess_property(data, target_size):
        # Scale the data to the range [0, 1]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        # Pad the data to the target size
        padded_data = pad_vector(scaled_data, target_size)

        return padded_data

    # Preprocess each property
    dominant_colors = df['dominant_color'].apply(
        lambda x: [int(color[i:i + 2], 16) for color in ast.literal_eval(x) for i in (1, 3, 5)]).tolist()
    image_widths = df['imagewidth'].to_numpy()
    image_heights = df['imageheight'].to_numpy()
    orientations = df['orientation'].to_numpy()
    makes = df['make'].apply(lambda x: sum(ord(c) for c in x)).to_numpy()
    tags = df['tags'].apply(lambda x: sum(ord(c) for word in x for c in word)).to_numpy()

    # Set target sizes for each property
    color_size = max_pad_size // 6
    width_size = max_pad_size // 6
    height_size = max_pad_size // 6
    orientation_size = max_pad_size // 6
    make_size = max_pad_size // 6
    tags_size = max_pad_size - 5 * (max_pad_size // 6)

    # Preprocess and pad each property
    colors_padded = np.vstack([preprocess_property(np.array(colors), color_size) for colors in dominant_colors])
    widths_padded = preprocess_property(image_widths, width_size)
    heights_padded = preprocess_property(image_heights, height_size)
    orientations_padded = preprocess_property(orientations, orientation_size)
    makes_padded = preprocess_property(makes, make_size)
    tags_padded = preprocess_property(tags, tags_size)

    # Concatenate the preprocessed properties to form a single vector
    vectors = np.hstack((
        np.ravel(colors_padded),
        np.ravel(widths_padded),
        np.ravel(heights_padded),
        np.ravel(orientations_padded),
        np.ravel(makes_padded),
        np.ravel(tags_padded)
    ))

    # Normalize the concatenated vector
    scaler = MinMaxScaler()
    normalized_vectors = scaler.fit_transform(vectors.reshape(-1, 1))

    # Pad the vector to the target size
    padded_vector = pad_vector(normalized_vectors, max_pad_size)

    # Convert the vector to a float32 array
    vector = padded_vector.astype(np.float32).ravel()

    logging.info("Vector: %s", vector)
    logging.info("Vector shape: %s", vector.shape)

    return vector


def process_new_metadata(new_metadata):
    logging.info("Processing new metadata %s", new_metadata)
    # Clean and preprocess new metadata to convert it to a vector
    cleaned_metadata = clean(new_metadata)
    # log cleaned metadata
    logging.info("Cleaned metadata: %s", cleaned_metadata)
    max_vector_size = 1024
    vector = preprocess_data(cleaned_metadata, max_vector_size)
    # log vector
    logging.info("Vector: %s\n Size of vector: %s", vector, len(vector))
    # if the vector size is not equal to max_vector_size, then add zeros to the end of the vector

    # pritn the size
    print("Size of vector: ", len(vector))
    # Insert the vector into the Milvus collection
    filename = new_metadata["filename"]
    collection_name = connect_to_milvus()
    collection = Collection(name=collection_name)

    logging.info("Inserting new metadata into Milvus")
    logging.info("Filename: %s", filename)

    #logging.info("Vector: %s", vector)

    insert_dataframe = pd.DataFrame({"filename": filename, "vector": [vector]})
    print(insert_dataframe)
    collection.insert(insert_dataframe)

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
        process_new_metadata(new_metadata)
