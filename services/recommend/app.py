import json
import logging
import os

import numpy as np
from flask import Flask, request, jsonify

from mysql.connector import pooling
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import spacy
import pandas as pd
from dotenv import load_dotenv

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


def hex_to_rgb(color):
    try:
        # remove the # from the color
        color = color[1:]
        # convert the color to rgb values
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        return rgb
    except:
        return 0, 0, 0


def get_metadata_from_mariadb_db():
    """
    Get the metadata from the MariaDB database

    :return: A pandas DataFrame with the metadata
    """
    # Open a connection to the database
    conn = connection_pool.get_connection()

    # Define the SQL query
    query = """
        SELECT filename, GROUP_CONCAT(CONCAT(mkey, '\t', mvalue) SEPARATOR '\n') AS metadata
        FROM metadata
        GROUP BY filename;
    """

    # Retrieve the metadata and create a DataFrame
    df = pd.read_sql(query, conn)

    # Close the connection
    conn.close()

    # Define the desired columns
    columns = ['filename', 'Make', 'ImageWidth', 'ImageHeight', 'Orientation', 'DateTimeOriginal',
               'dominant_color', 'tags']

    # Create an empty DataFrame with the desired columns
    metadata_df = pd.DataFrame(columns=columns)

    # Fill the DataFrame with the metadata
    for _, row in tqdm(df.iterrows(), desc="Get metadata from database", total=df.shape[0]):
        try:
            props = {'filename': row['filename']}
            metadata_str = row['metadata'].split('\n')
            for prop in metadata_str:
                if prop:
                    k, value = prop.split('\t')
                    if k in columns[1:]:
                        if k == 'dominant_color':
                            color_list = eval(value)
                            color_list = [c[0] for c in color_list]
                            props[k] = color_list
                        elif k == 'tags':
                            props[k] = eval(value)
                        else:
                            props[k] = value

            metadata_df = metadata_df.append(props, ignore_index=True)
        except Exception as e:
            continue

    # Set missing columns to None
    for col in columns:
        if col not in metadata_df.columns:
            metadata_df[col] = np.nan

    return metadata_df


def get_clean_preferences(df_preferences):
    # Replace all NaN values with empty strings with the fillna() method
    df_preferences = df_preferences.fillna(0)
    # remove the rows with nan in dominant_color
    if 'dominant_color' in df_preferences.columns:
        df_preferences = df_preferences.dropna(subset=['dominant_color'])
        # convert colors to rgb values
        df_preferences['dominant_color'] = df_preferences['dominant_color'].apply(lambda x: hex_to_rgb(x))
        # replace all 0 values with empty strings
        df_preferences['dominant_color'] = df_preferences['dominant_color'].replace(0, '')

    return df_preferences


def get_clean_dataset():
    try:
        metadata = get_metadata_from_mariadb_db()
    except Exception as e:
        print(f"An error occurred while fetching metadata: {e}")
        return None
    # convert to DataFrame
    df_metadata = pd.DataFrame(metadata)
    # remove the rows with nan in dominant_color
    df_metadata = df_metadata.dropna(subset=['dominant_color'])
    # split dominant color into 4 columns and remove the dominant_color column
    if 'dominant_color' in df_metadata.columns:
        df_metadata['color1'] = df_metadata['dominant_color'].apply(lambda x: x[0] if len(x) >= 1 else 0)
        df_metadata['color2'] = df_metadata['dominant_color'].apply(lambda x: x[1] if len(x) >= 2 else 0)
        df_metadata['color3'] = df_metadata['dominant_color'].apply(lambda x: x[2] if len(x) == 3 else 0)
        df_metadata['color4'] = df_metadata['dominant_color'].apply(lambda x: x[3] if len(x) == 4 else 0)
        # convert colors to rgb values
        df_metadata['color1'] = df_metadata['color1'].apply(lambda x: hex_to_rgb(x) if x else (0, 0, 0))
        df_metadata['color2'] = df_metadata['color2'].apply(lambda x: hex_to_rgb(x) if x else (0, 0, 0))
        df_metadata['color3'] = df_metadata['color3'].apply(lambda x: hex_to_rgb(x) if x else (0, 0, 0))
        df_metadata['color4'] = df_metadata['color4'].apply(lambda x: hex_to_rgb(x) if x else (0, 0, 0))
        df_metadata = df_metadata.drop('dominant_color', axis=1)
    else:
        df_metadata['color1'] = 0
        df_metadata['color2'] = 0
        df_metadata['color3'] = 0
        df_metadata['color4'] = 0

    # convert the tags column to a list of strings
    df_metadata = df_metadata.fillna(0)
    # remove all columns except filename, tags, color1, color2, color3, color4, Make, Width, Height
    df_metadata = df_metadata[
        ['filename', 'Make', 'ImageWidth', 'ImageHeight', 'Orientation', 'DateTimeOriginal', 'tags', 'color1', 'color2',
         'color3', 'color4']]
    # replace all 0 values with empty strings
    df_metadata['Make'] = df_metadata['Make'].replace(0, '')

    return df_metadata


def recommend_colors(df_metadata, df_preferences, n=0):
    # Load the dataset into a Pandas DataFrame
    data = df_metadata.copy()

    # Extract the individual r, g, and b values from tupbles in the color columns
    data[['r1', 'g1', 'b1']] = pd.DataFrame(data['color1'].tolist(), index=data.index)
    data[['r2', 'g2', 'b2']] = pd.DataFrame(data['color2'].tolist(), index=data.index)
    data[['r3', 'g3', 'b3']] = pd.DataFrame(data['color3'].tolist(), index=data.index)
    data[['r4', 'g4', 'b4']] = pd.DataFrame(data['color4'].tolist(), index=data.index)

    # Normalize the r, g, and b columns to be between 0 and 1
    data[['r1', 'g1', 'b1', 'r2', 'g2', 'b2', 'r3', 'g3', 'b3', 'r4', 'g4', 'b4']] = data[['r1', 'g1', 'b1', 'r2', 'g2',
                                                                                           'b2', 'r3', 'g3', 'b3', 'r4',
                                                                                           'g4', 'b4']] / 255

    # Normalize the input RGB color to be between 0 and 1
    r, g, b = df_preferences['dominant_color'][0]
    r_norm, g_norm, b_norm = r / 255, g / 255, b / 255

    # Compute the Euclidean distance between the input color and all the colors in the dataset
    data['similarity_dominant_color'] = euclidean_distances(
        [[r_norm, g_norm, b_norm, r_norm, g_norm, b_norm, r_norm, g_norm, b_norm, r_norm, g_norm, b_norm]],
        data[['r1', 'g1', 'b1', 'r2', 'g2', 'b2', 'r3', 'g3', 'b3', 'r4', 'g4', 'b4']])[0]

    # Sort the dataset by Euclidean distance in ascending order and return the top 10 closest matches
    if n == 0:
        closest_matches = data.sort_values('similarity_dominant_color', ascending=True)[
            ['filename', 'color1', 'color2', 'color3', 'color4', 'similarity_dominant_color']]
    else:
        closest_matches = data.sort_values('similarity_dominant_color', ascending=True).head(n)[
            ['filename', 'color1', 'color2', 'color3', 'color4', 'similarity_dominant_color']]

    return closest_matches


def recommend_tags(df_metadata, df_preferences, n=0, nlp=None):
    # Load the spaCy model if it hasn't been loaded
    if not nlp:
        # download the model
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")

    # Define the preferences list and the dataframe
    preferences = df_preferences['tags'][0]
    # Load dataset with words and drop duplicate rows
    df = df_metadata.copy()
    df = df.dropna(subset=["tags"]).reset_index(drop=True)
    # replace int with empty list
    df['tags'] = df['tags'].apply(lambda x: x if x else [])

    preferences = [word.strip() for word in preferences.strip('[]').split(',')]

    # PROBLEM HERE

    # Precompute the similarity between each tag word and each preference word
    similarity_dict = {}
    for tag_word in set([word for tags in df['tags'] for word in tags]):
        for pref_word in preferences:
            similarity_dict[(tag_word, pref_word)] = nlp(tag_word).similarity(nlp(pref_word))

    # Compute the average similarity for each row in the dataframe
    similarities = []
    for tags in df['tags']:
        sum_similarity = 0
        for tag_word in tags:
            for pref_word in preferences:
                sum_similarity += similarity_dict[(tag_word, pref_word)]
        avg_similarity = sum_similarity / (len(tags) * len(preferences)) if len(tags) > 0 else 0
        similarities.append(avg_similarity)

    # Add the similarity scores to a new column in the dataframe
    df['similarity_tags'] = similarities
    if n == 0:
        closest_matches = df.sort_values('similarity_tags', ascending=False)[
            ['filename', 'similarity_tags']]
    else:
        closest_matches = df.sort_values('similarity_tags', ascending=False).head(n)[
            ['filename', 'similarity_tags']]

    return closest_matches


def recommend_make(df_metadata, df_preferences, n=0):
    # Load the spaCy model
    nlp = spacy.load("en_core_web_md")

    # Define the preferences list and the dataframe
    make = df_preferences['Make'][0]
    # Load dataset with words and drop duplicate rows
    df = df_metadata.copy()
    df = df.dropna(subset=["Make"]).reset_index(drop=True)

    # Convert make and Make to document objects
    make_doc = nlp(make)
    df['Make'] = df['Make'].apply(nlp)

    # Compute the cosine similarity between the make preferences and all the makes in the dataset
    similarities = [make_doc.similarity(doc) for doc in df['Make']]

    # Add the similarity scores to a new column in the dataframe
    df['similarity_make'] = similarities
    if n == 0:
        closest_matches = df.sort_values('similarity_make', ascending=False)[
            ['filename', 'similarity_make']]
    else:
        closest_matches = df.sort_values('similarity_make', ascending=False).head(n)[
            ['filename', 'similarity_make']]

    return closest_matches


def recommend_orientation(df_metadata, df_preferences, n=0):
    # Define the preferences list and the dataframe
    orientation = df_preferences['Orientation'][0]
    # Load dataset with words and drop duplicate rows
    df = df_metadata.dropna(subset=["Orientation"]).reset_index(drop=True)
    # if Orientation contain '' or '0' or '1' then replace with 0 or 1
    df['Orientation'] = df['Orientation'].apply(lambda x: 0 if x == '' or x == '0' else 1)

    # Convert the Orientation column to integer type
    df['Orientation'] = df['Orientation'].astype(int)

    # Orientation is 0 or 1, so we can just subtract the preference from the orientation
    df['similarity_orientation'] = df['Orientation'].apply(lambda x: abs(x - orientation))

    # sort by similarity
    if n > 0:
        closest_matches = df.sort_values('similarity_orientation', ascending=False).head(n)[
            ['filename', 'similarity_orientation']]
    else:
        closest_matches = df.sort_values('similarity_orientation', ascending=False)[
            ['filename', 'similarity_orientation']]

    return closest_matches


def recommend_size(df_metadata, df_preferences, n=0):
    # Define the preferences list and the dataframe
    width = int(df_preferences['ImageWidth'][0])
    height = int(df_preferences['ImageHeight'][0])
    # Load dataset with words and drop duplicate rows
    df = df_metadata.dropna(subset=["ImageWidth", "ImageHeight"]).reset_index(drop=True)

    # Convert the ImageWidth and ImageHeight column to integer type
    df[['ImageWidth', 'ImageHeight']] = df[['ImageWidth', 'ImageHeight']].astype(int)

    # Compute the product of width and height outside the loop
    product = width * height

    # Use apply method to compute similarity score for each row
    df['similarity_size'] = df.apply(lambda x: 1 - abs(product - (x['ImageWidth'] * x['ImageHeight'])) / product,
                                     axis=1)

    if n == 0:
        closest_matches = df.sort_values('similarity_size', ascending=False)[
            ['filename', 'similarity_size']]
    else:
        closest_matches = df.sort_values('similarity_size', ascending=False).head(n)[
            ['filename', 'similarity_size']]

    return closest_matches


def recommend(df_metadata, df_preferences, n=0):
    # Create empty dataframe to store similarity scores
    similarities_df = pd.DataFrame(columns=['filename'])
    # fill filename column with filenames
    similarities_df['filename'] = df_metadata['filename']

    # Dictionary to map preference names to similarity column names
    similarity_columns = {
        'dominant_color': 'similarity_dominant_color',
        'tags': 'similarity_tags',
        'Make': 'similarity_make',
        'Orientation': 'similarity_orientation',
        'ImageWidth': 'similarity_size',
        'ImageHeight': 'similarity_size'
    }

    # Iterate over each preference and call the corresponding recommendation method
    for preference in df_preferences.columns:
        if preference == 'dominant_color':
            # Call recommend_colors method
            similarity_scores = recommend_colors(df_metadata, df_preferences, n)
            similarities_df = similarities_df.merge(similarity_scores, on='filename', how='left')
        elif preference == 'tags':
            # Call recommend_tags method
            similarity_scores = recommend_tags(df_metadata, df_preferences, n)
            similarities_df = similarities_df.merge(similarity_scores, on='filename', how='left')
        elif preference == 'Make':
            # Call recommend_make method
            similarity_scores = recommend_make(df_metadata, df_preferences, n)
            similarities_df = similarities_df.merge(similarity_scores, on='filename', how='left')
        elif preference == 'Orientation':
            # Call recommend_orientation method
            similarity_scores = recommend_orientation(df_metadata, df_preferences, n)
            similarities_df = similarities_df.merge(similarity_scores, on='filename', how='left')
        elif preference in ['ImageWidth', 'ImageHeight']:
            # Call recommend_size method if one of the size is not given, then skip this preference and verify that
            # similarity_df do not contain this preference
            if df_preferences['ImageWidth'][0] == ''\
                    or df_preferences['ImageHeight'][0] == ''\
                    or 'similarity_size' in similarities_df.columns:
                continue
            # append the similarity scores to the dataframe
            similarity_scores = recommend_size(df_metadata, df_preferences, n)
            similarities_df = similarities_df.merge(similarity_scores, on='filename', how='left')
        else:
            raise ValueError(f"Invalid preference: {preference}")

    # Compute weighted average of similarity scores for each row
    weights = {
        'dominant_color': 0.2,
        'tags': 0.3,
        'Make': 0.2,
        'Orientation': 0.1,
        'size': 0.2
    }

    # Compute total weight for all preferences provided by the user but not size add it after the sum cause the key
    # size is not in the df_preferences.columns
    total_weight = sum([weights[preference] for preference in df_preferences.columns if preference != 'ImageWidth' and preference != 'ImageHeight'])
    # Add the size weight to the total weight
    total_weight += weights['size']

    # Update weights based on the total weight
    if len(df_preferences.columns) == 1:
        # If only one preference is provided and it's not size, set the weight for that preference to 1
        weights[df_preferences.columns[0]] = 1.0
    else:
        # Compute the weight for each preference based on the total weight
        for preference in df_preferences.columns:
            if preference in ['ImageWidth', 'ImageHeight']:
                continue
            else:
                weights[preference] = weights[preference] + ((1 - total_weight) * (weights[preference] / total_weight))

    # Check the sum of all used weights Compute total weight for all preferences provided by the user but not size
    # add it after the sum cause the key size is not in the df_preferences.columns
    total_weight = sum([weights[preference] for preference in df_preferences.columns if preference != 'ImageWidth' and preference != 'ImageHeight'])
    # Add the size weight to the total weight
    total_weight += weights['size']
    if total_weight != 1.0:
        # update size weight
        weights['size'] = 1.0 - total_weight + weights['size']

    # The score is between 0 and 1, so we can just multiply the score by the weight and sum them up
    similarities_df['similarity_total'] = 0.0
    for preference in df_preferences.columns:
        if preference in ['ImageWidth', 'ImageHeight']:
            # if one of the size is not given, then skip this preference
            if df_preferences['ImageWidth'][0] == '' or df_preferences['ImageHeight'][0] == '':
                similarities_df['similarity_total'] += weights['size'] * similarities_df['similarity_size']
        else:
            similarities_df['similarity_total'] += weights[preference] * similarities_df[similarity_columns[preference]]

    # Sort dataframe by similarity score in descending order and return top n matches
    if n == 0:
        closest_matches = similarities_df.sort_values('similarity_total', ascending=False)[
            ['filename', 'similarity_total']]
    else:
        closest_matches = similarities_df.sort_values('similarity_total', ascending=False).head(n)[
            ['filename', 'similarity_total']]

    return closest_matches


@app.route('/recommend', methods=['POST'])
def recommend_images():
    # get the preferences from the request
    preferences = request.get_json()
    # valid preferences are Make, ImageWidth, ImageHeight, Orientation, dominant_color, tags
    preferences = {k: v for k, v in preferences.items() if k in
                   ['Make', 'ImageWidth', 'ImageHeight', 'Orientation', 'dominant_color', 'tags']
                   }

    # If there is ImageWidth is set, but ImageHeight is not, set ImageHeight to ImageWidth
    if 'ImageWidth' in preferences and 'ImageHeight' not in preferences:
        preferences['ImageHeight'] = preferences['ImageWidth']
    # If there is ImageHeight is sxet, but ImageWidth is not, set ImageWidth to ImageHeight
    if 'ImageHeight' in preferences and 'ImageWidth' not in preferences:
        preferences['ImageWidth'] = preferences['ImageHeight']

    # Not sure about this line, but it seems to work
    update_preferences = {k: [v] for k, v in preferences.items()}

    df_metadata = get_clean_dataset()
    df_pref = pd.DataFrame(update_preferences)
    df_preferences = get_clean_preferences(df_pref)

    result = recommend(df_metadata, df_preferences)
    # get the top 10 results and return them as base64 encoded images
    top_10 = pd.DataFrame(result.head(10))
    return top_10.to_json(orient='records')


# TEST CODE
from sklearn.metrics.pairwise import cosine_similarity
from collections import namedtuple
import numpy as np

ImageProperties = namedtuple('ImageProperties', ['name', 'hex_color', 'tags', 'make', 'orientation', 'width', 'height'])

# Sample data
data = [
    ImageProperties('Image1', '#FF0000', ['car', 'road'], 'Canon', 0, 1024, 768),
    ImageProperties('Image2', '#00FF00', ['flower', 'garden'], 'Nikon', 1, 800, 600),
    # Add more images...
]

def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

def preprocess_data(data):
    for image in data:
        image_tags = " ".join(image.tags)
        yield np.array([
            *hex_to_rgb(image.hex_color),
            len(image_tags),
            len(image.make),
            image.orientation,
            image.width,
            image.height
        ])



def test_recommend_images(image_index, data, top_n=10):
    preprocessed_data = np.array(list(preprocess_data(data)))
    similarity_matrix = cosine_similarity(preprocessed_data)
    most_similar_indices = np.argsort(-similarity_matrix[image_index])[1:top_n+1]
    return [data[i] for i in most_similar_indices]


@app.route("/test", methods=['GET'])
def test():
    # Test the recommender system
    recommended_images = test_recommend_images(0, data)
    return jsonify([image.name for image in recommended_images])



if __name__ == '__main__':
    app.run(debug=False)
