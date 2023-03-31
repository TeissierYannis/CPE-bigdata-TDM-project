import os

from flask import Flask, request

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
    # Create a cursor
    c = conn.cursor()

    # Retrieve the metadata
    c.execute("""
        SELECT filename, GROUP_CONCAT(CONCAT(mkey, '\t', mvalue) SEPARATOR '\n') AS metadata
        FROM metadata
        GROUP BY filename;
    """)
    metadata = c.fetchall()

    # Close the connection
    conn.close()

    # Create an empty DataFrame with the desired columns
    columns = ['filename', 'Make', 'ImageWidth', 'ImageHeight', 'Orientation', 'DateTimeOriginal',
               'dominant_color', 'tags']
    df = pd.DataFrame(columns=columns)

    # Fill the DataFrame with the metadata
    for image in tqdm(metadata, desc="Get metadata from database"):
        try:
            props = {'filename': image[0]}
            metadata_str = image[1].split('\n')
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
            df = df.append(props, ignore_index=True)
        except Exception as e:
            return e


    # Verify that df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df should be a pandas DataFrame")

    # Verify that df has the correct columns
    if not set(columns).issubset(df.columns):
        missing_columns = set(columns) - set(df.columns)
        raise ValueError(f"Missing columns: {missing_columns}")

    # Verify that df has the correct shape
    if df.shape[0] < 1:
        raise ValueError("df should have at least 1 row")

    return df


def get_clean_preferences(df_preferences):
    # remove the rows with nan in dominant_color
    df_preferences = df_preferences.dropna(subset=['dominant_color'])
    # split dominant color into 4 columns and remove the dominant_color column
    # convert the tags column to a list of strings
    # Replace all NaN values with empty strings with the fillna() method
    df_preferences = df_preferences.fillna(0)
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

    # Precompute the similarity between each tag word and each preference word
    similarity_dict = {}
    for tag_word in set([word for tags in df['tags'] for word in tags]):
        for pref_word in set(preferences):
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
    df['similarity_size'] = df.apply(lambda x: 1 - abs(product - (x['ImageWidth'] * x['ImageHeight'])) / product, axis=1)

    if n == 0:
        closest_matches = df.sort_values('similarity_size', ascending=False)[
            ['filename', 'similarity_size']]
    else:
        closest_matches = df.sort_values('similarity_size', ascending=False).head(n)[
            ['filename', 'similarity_size']]

    return closest_matches


def recommend(df_metadata, df_preferences, n=0):
    # Assign weights to properties based on user preferences
    weights = {
        'Make': float(5.0),
        'ImageWidth': float(1.0),
        'ImageHeight': float(1.0),
        'Orientation': float(2.0),
        'dominant_color': float(3.0),
        'tags': float(5.0)
    }

    # Create a dictionary with the preferences and the corresponding recommendation methods
    preference_methods = {
        'Make': recommend_make,
        'ImageWidth': recommend_size,
        'ImageHeight': recommend_size,
        'Orientation': recommend_orientation,
        'dominant_color': recommend_colors,
        'tags': recommend_tags
    }

    # Remove preferences with no values
    preferences = {k: v for k, v in df_preferences.squeeze().to_dict().items() if v != ''}

    # Calculate the sum of the weights
    weights_sum = 0
    for key in weights:
        weights_sum += weights[key]
    for key in weights:
        weights[key] = weights[key] / weights_sum

    # Calculate similarity score for each property
    df_metadata['similarity_score'] = 0.0
    for preference, value in preferences.items():
        method = preference_methods[preference]
        # if it's height or width, the name of the column is different similarity_size
        if preference == 'ImageWidth' or preference == 'ImageHeight':
            similarity = method(df_metadata, df_preferences, n)['similarity_size'].astype(float)
        else:
            similarity = method(df_metadata, df_preferences, n)[f'similarity_{preference.lower()}'].astype(float)
        df_metadata['similarity_score'] += similarity * (weights[preference] / weights_sum)

    # Replace NaN values in the 'similarity_score' column with 0
    df_metadata['similarity_score'].fillna(0, inplace=True)

    # Sort by similarity score
    if n == 0:
        closest_matches = df_metadata.sort_values('similarity_score', ascending=False)[
            ['filename', 'similarity_score']]
    else:
        closest_matches = df_metadata.sort_values('similarity_score', ascending=False).head(n)[
            ['filename', 'similarity_score']]

    return closest_matches


@app.route('/recommend', methods=['POST'])
def recommend_images():
    # get the preferences from the request
    preferences = request.get_json()
    # valid preferences are Make, ImageWidth, ImageHeight, Orientation, dominant_color, tags
    preferences = {k: v for k, v in preferences.items() if k in ['Make', 'ImageWidth', 'ImageHeight', 'Orientation', 'dominant_color', 'tags']}
    #preferences = {
    #   'Make': '',
    #   'ImageWidth': '',
    #   'ImageHeight': '',
    #   'Orientation': 1,
    #   'dominant_color': '#73AD3D',
    #   'tags': ['vase', 'toilet']
    #}

    df_metadata = get_clean_dataset()
    df_pref = pd.DataFrame([preferences])
    df_preferences = get_clean_preferences(df_pref)
    result = recommend(df_metadata, df_preferences)
    # get the top 10 results and return them as base64 encoded images
    top_10 = pd.DataFrame(result.head(10))
    return top_10.to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=False)
