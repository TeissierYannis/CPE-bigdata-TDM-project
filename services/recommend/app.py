import os

from flask import Flask, jsonify, request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mysql.connector import pooling
from dotenv import load_dotenv
from collections import namedtuple
import ast

load_dotenv()

app = Flask(__name__)

cnf = {
    'host': os.getenv("SQL_HOST") or 'localhost',
    'user': os.getenv("SQL_USER") or 'root',
    'port': os.getenv("SQL_PORT") or 3306,
    'password': os.getenv("SQL_PASSWORD") or '',
    'database': os.getenv("SQL_DATABASE") or 'harvest'
}

# Create a connection pool
connection_pool = pooling.MySQLConnectionPool(pool_name="mypool",
                                              pool_size=2,
                                              **cnf)

ImageProperties = namedtuple(
    'ImageProperties', ['name', 'hex_color', 'tags', 'make', 'orientation', 'width', 'height', 'distance']
)

UserPreferences = namedtuple(
    'UserPreferences', ['dominant_color', 'tags', 'Make', 'Orientation', 'ImageWidth', 'ImageHeight']
)


def get_metadata_from_mariadb_as_imageproperties():
    # Open a connection to the database
    conn = connection_pool.get_connection()
    # Create a cursor
    c = conn.cursor()

    # Retrieve the metadata
    c.execute("""
        SELECT DISTINCT filename, GROUP_CONCAT(CONCAT(mkey, '\t', mvalue) SEPARATOR '\n') AS metadata
        FROM metadata
        WHERE mkey IN ('Make', 'Orientation', 'ImageWidth', 'ImageHeight', 'tags', 'dominant_color')
        GROUP BY filename;
    """)
    metadata = c.fetchall()

    # Close the connection
    conn.close()

    # use the namedtuple ImageProperties to store the metadata
    images = []

    # Loop through the rows of metadata
    for row in metadata:
        try:
            filename, metadata_str = row
            metadata_items = metadata_str.split('\n')
            metadata_dict = {key: value for key, value in (item.split('\t') for item in metadata_items)}

            # Clean dominant colors: convert the string to a list of tuples and extract only the color hex codes
            dominant_colors = ast.literal_eval(metadata_dict.get('dominant_color', '[]'))
            hex_colors = [color[0] for color in dominant_colors]

            # Clean tags: convert the string to a list of strings
            tags = ast.literal_eval(metadata_dict.get('tags', '[]'))

            # Create an ImageProperties object for each row
            image = ImageProperties(
                name=filename,
                hex_color=hex_colors,
                tags=tags,
                make=metadata_dict.get('Make', None),
                orientation=metadata_dict.get('Orientation', None),
                width=metadata_dict.get('ImageWidth', None),
                height=metadata_dict.get('ImageHeight', None),
                distance=0
            )

            # Add the ImageProperties object to the list
            images.append(image)
        except:
            pass

    return images


def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def preprocess_data(data):
    for image in data:
        image_tags = " ".join(image.tags)
        avg_rgb_color = np.mean([hex_to_rgb(color) for color in image.hex_color], axis=0)
        try:
            make_len = len(image.make)
        except TypeError:
            make_len = 0

        try:
            width = int(image.width)
        except TypeError:
            width = 0

        try:
            height = int(image.height)
        except TypeError:
            height = 0

        try:
            orientation = int(image.orientation)
        except:
            orientation = 0

        avg_rgb_color_list = avg_rgb_color.tolist() if hasattr(avg_rgb_color, 'tolist') else [avg_rgb_color]

        try:
            yield np.array([
                *avg_rgb_color_list,  # Ensure avg_rgb_color is an iterable before unpacking
                len(image_tags),
                make_len,
                orientation,
                width,
                height,
                0
            ])
        except:
            pass


def user_preferences_vector(preferences: UserPreferences):
    preference_vector = []

    if preferences.dominant_color:
        try:
            dominant_color = hex_to_rgb(preferences.dominant_color)
        except:
            dominant_color = (255, 255, 255)
        preference_vector.extend(dominant_color)

    if preferences.tags:
        tags = " ".join(preferences.tags)
        preference_vector.append(len(tags))

    if preferences.Make:
        make_len = len(preferences.Make)
        preference_vector.append(make_len)

    if preferences.Orientation is not None:
        try:
            orientation = int(preferences.Orientation)
        except:
            orientation = 0
        preference_vector.append(orientation)

    if preferences.ImageWidth is not None:
        try:
            width = int(preferences.ImageWidth)
        except:
            width = 0
        preference_vector.append(width)

    if preferences.ImageHeight is not None:
        try:
            height = int(preferences.ImageHeight)
        except:
            height = 0
        preference_vector.append(height)

    return np.array(preference_vector)


def recommend_images(preferences, data, top_n=0):
    preprocessed_data = np.array(list(preprocess_data(data)))
    user_vector = user_preferences_vector(preferences)

    preference_mask = create_preference_mask(preferences)
    similarity_matrix = masked_cosine_similarity(user_vector, preprocessed_data, preference_mask)

    # if top_n is 0, return all images
    if top_n == 0:
        top_n = len(data)

    for i, image in enumerate(data):
        data[i] = image._replace(distance=similarity_matrix[0][i + 1])

    # sort the data by distance
    data.sort(key=lambda x: x.distance, reverse=True)

    # return the top_n images
    return data[:top_n]


def create_preference_mask(preferences: UserPreferences):
    mask = []
    if preferences.dominant_color:
        mask.extend([True] * 3)  # RGB
    if preferences.tags:
        mask.append(True)
    if preferences.Make:
        mask.append(True)
    if preferences.Orientation is not None:
        mask.append(True)
    if preferences.ImageWidth is not None and preferences.ImageHeight is not None:
        mask.extend([True] * 2)  # Width and Height
    return mask

def masked_cosine_similarity(user_vector, data, mask):
    masked_user_vector = user_vector[mask]
    masked_data = data[:, mask]
    return cosine_similarity(np.vstack([masked_user_vector, masked_data]))


@app.route("/recommend", methods=['GET'])
def test():
    data = get_metadata_from_mariadb_as_imageproperties()

    # Get preferences from the user
    preferences = request.get_json()

    # Validate the preferences
    if not preferences:
        return jsonify({'error': 'No preferences provided'}), 400

    # get the preferences from the request
    preferences = request.get_json()
    # valid preferences are Make, ImageWidth, ImageHeight, Orientation, dominant_color, tags
    user_preference = UserPreferences(
        Make=preferences.get('Make', None),
        ImageWidth=preferences.get('ImageWidth', None),
        ImageHeight=preferences.get('ImageHeight', None),
        Orientation=preferences.get('Orientation', None),
        dominant_color=preferences.get('dominant_color', None),
        tags=preferences.get('tags', None)
    )

    # Check if there is only one width or height and if yes, set the width = to the height or vice versa
    if user_preference.ImageWidth and not user_preference.ImageHeight:
        user_preference.ImageHeight = user_preference.ImageWidth
    elif user_preference.ImageHeight and not user_preference.ImageWidth:
        user_preference.ImageWidth = user_preference.ImageHeight

    # Test the recommender system
    recommended_images = recommend_images(user_preference, data, top_n=10)
    return jsonify(recommended_images)


if __name__ == '__main__':
    app.run(debug=False)
