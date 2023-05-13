import os
import io
import ast
import spacy
import folium
import datetime
import squarify
import webcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from collections import Counter
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from sqlalchemy.orm import sessionmaker
from scipy.spatial.distance import pdist
from sqlalchemy import create_engine, text
from flask import Flask, Response, request, send_file, jsonify
from scipy.cluster.hierarchy import dendrogram, linkage

load_dotenv()

app = Flask(__name__)


def get_metadata_from_postgres_db():
    """
    Get the metadata from the PostgreSQL database

    :param db_name: The name of the database
    :param user: The username to connect to the database
    :param password: The password to connect to the database
    :param host: The hostname or IP address of the database server
    :param port: The port number to connect to the database server
    :return: A dictionary with the metadata
    """
    print("Connecting to database...")

    # Create the database engine
    engine = create_engine("postgresql://postgres:postgres@postgres:5432/raw_metadata")

    # Create a session
    Session = sessionmaker(bind=engine)
    session = Session()

    print("Retrieving metadata from database...")

    # Retrieve the metadata
    result = session.execute(
        text("""SELECT filename, Make, Model, Software, BitsPerSample, ImageWidth, ImageHeight, ImageDescription,
    Orientation, Copyright, DateTime, DateTimeOriginal, DateTimeDigitized, SubSecTimeOriginal,
    ExposureTime, FNumber, ExposureProgram, ISOSpeedRatings, SubjectDistance, ExposureBiasValue,
    Flash, FlashReturnedLight, FlashMode, MeteringMode, FocalLength, FocalLengthIn35mm,
    Latitude, LatitudeDegrees, LatitudeMinutes, LatitudeSeconds, LatitudeDirection,
    Longitude, LongitudeDegrees, LongitudeMinutes, LongitudeSeconds, LongitudeDirection,
    Altitude, DOP, FocalLengthMin, FocalLengthMax, FStopMin, FStopMax, LensMake, LensModel,
    FocalPlaneXResolution, FocalPlaneYResolution, tags, dominant_color
    FROM image_metadata;""")
    )

    keys = [
        'filename', 'Make', 'Model', 'Software', 'BitsPerSample', 'ImageWidth', 'ImageHeight', 'ImageDescription',
        'Orientation', 'Copyright', 'DateTime', 'DateTimeOriginal', 'DateTimeDigitized', 'SubSecTimeOriginal',
        'ExposureTime', 'FNumber', 'ExposureProgram', 'ISOSpeedRatings', 'SubjectDistance', 'ExposureBiasValue',
        'Flash', 'FlashReturnedLight', 'FlashMode', 'MeteringMode', 'FocalLength', 'FocalLengthIn35mm',
        'Latitude', 'LatitudeDegrees', 'LatitudeMinutes', 'LatitudeSeconds', 'LatitudeDirection',
        'Longitude', 'LongitudeDegrees', 'LongitudeMinutes', 'LongitudeSeconds', 'LongitudeDirection',
        'Altitude', 'DOP', 'FocalLengthMin', 'FocalLengthMax', 'FStopMin', 'FStopMax', 'LensMake', 'LensModel',
        'FocalPlaneXResolution', 'FocalPlaneYResolution', 'tags', 'dominant_color'
    ]

    # Convert the metadata to a dictionary
    metadata_dict = {}
    for row in tqdm(result, desc="Get metadata from database"):
        try:
            # use row to find
            metadata_dict[row[0]] = {}
            for i in range(len(keys)):
                metadata_dict[row[0]][keys[i]] = row[i]
        except Exception as e:
            print(e, row)

    # Close the session
    session.close()

    return metadata_dict


def dms_to_decimal(degrees, minutes, seconds):
    """
    Convert DMS (degrees, minutes, seconds) coordinates to DD (decimal degrees)
    :param degrees: degrees
    :param minutes: minutes
    :param seconds:  seconds
    :return: decimal coordinates
    """
    decimal_degrees = abs(degrees) + (minutes / 60) + (seconds / 3600)

    if degrees < 0:
        decimal_degrees = -decimal_degrees

    return decimal_degrees


def clean_gps_infos(metadata_to_cln):
    """
    Clean the GPS infos

    :param metadata_to_cln: The metadata to clean
    :return: A dictionary with the cleaned metadata
    """

    cpt_valid, cpt_invalid, cpt_converted = 0, 0, 0
    for file in tqdm(metadata_to_cln, desc="Clean GPS values"):
        file_meta = metadata_to_cln[file]

        if 'Latitude' in file_meta:
            has_dms_values = file_meta['LatitudeDegrees'] != '0.000000' or file_meta['LongitudeDegrees'] != '0.000000'
            has_decimal_values = file_meta['Latitude'] != '0.000000' or file_meta['Longitude'] != '0.000000'

            if has_dms_values or has_decimal_values:
                try:
                    should_convert = '.' not in file_meta['Latitude'] and has_dms_values
                except:
                    continue

                if should_convert:
                    # calculate the decimal coordinates from the degrees coordinates
                    latitude = dms_to_decimal(
                        float(file_meta['LatitudeDegrees']),
                        float(file_meta['LatitudeMinutes']),
                        float(file_meta['LatitudeSeconds']))

                    longitude = dms_to_decimal(
                        float(file_meta['LongitudeDegrees']),
                        float(file_meta['LongitudeMinutes']),
                        float(file_meta['LongitudeSeconds']))

                    cpt_converted += 1
                else:
                    # convert the coordinates to float
                    latitude = float(file_meta['Latitude'])
                    longitude = float(file_meta['Longitude'])

                # update the metadata with the calculated latitude and longitude
                metadata_to_cln[file]['Latitude'] = latitude
                metadata_to_cln[file]['Longitude'] = longitude
                cpt_valid += 1

            else:
                metadata_to_cln[file]['Latitude'] = None
                metadata_to_cln[file]['Longitude'] = None
                metadata_to_cln[file]['Altitude'] = None
                cpt_invalid += 1

    print("GPS values : \n",
          "Valid : ", cpt_valid,
          "\nInvalid : ", cpt_invalid,
          "\nConverted : ", cpt_converted,
          )

    return metadata_to_cln


def clean_metadata(metadata_to_clean):
    """
    Clean the metadata
    Remove special characters from the 'Make' property values
    Remove the 'T' and '-' characters from the 'DateTime' property values

    :param metadata_to_clean: The metadata to clean
    :return: A dictionary with the cleaned metadata
    """
    cln_meta = metadata_to_clean.copy()

    # Clean 'Make' property values
    try:
        for file in tqdm(cln_meta, desc="Clean 'Make' property values"):
            if 'Make' in cln_meta[file]:
                cln_meta[file]['Make'] = ''.join(filter(str.isalpha, cln_meta[file]['Make'])).replace('CORPORATION',
                                                                                                      '').replace(
                    'CORP', '').replace('COMPANY', '').replace('LTD', '').replace('IMAGING', '')
    except Exception as e:
        print(e)

    # Clean 'DateTime' property values
    cpt, cpt_error = 0, 0
    date_error = []
    try:

        for file in tqdm(cln_meta, desc="Clean 'DateTime' property values"):
            if 'DateTimeOriginal' in cln_meta[file]:
                date = cln_meta[file]['DateTimeOriginal']
                try:
                    if date is not None:
                        tmp = date.replace('T', ' ').replace('-', ':').split('+')[0]
                        cln_meta[file]['DateTimeOriginal'] = datetime.datetime.strptime(tmp[:19], '%Y:%m:%d %H:%M:%S')
                        # if the year is after actual year, we assume that the date is wrong
                        if cln_meta[file]['DateTimeOriginal'].year > datetime.datetime.now().year:
                            date_error.append(cln_meta[file]['DateTimeOriginal'])
                            cln_meta[file]['DateTimeOriginal'] = None
                            cpt_error += 1
                        else:
                            cpt += 1
                except ValueError:
                    date_error.append(date)
                    cln_meta[file]['DateTimeOriginal'] = None
                    cpt_error += 1
    except Exception as e:
        print(e)

    print(f"Metadata cleaned ! {cpt}/{len(cln_meta)} dates OK, {cpt_error} dates KO")
    print(f"Dates KO : {date_error}")

    # Clean 'tags' property values
    for file in tqdm(cln_meta, desc="Clean 'tags' property values"):
        if 'tags' in cln_meta[file]:
            val = None
            if cln_meta[file]['tags'] is not None:
                val = eval(cln_meta[file]['tags'])
            cln_meta[file]['tags'] = val

    # Clean the GPS infos
    cln_meta = clean_gps_infos(cln_meta)

    return cln_meta


def get_metadata():
    """
    Get the metadata from the database
    :return: A JSON object with the metadata
    """
    # Check if the metadata file already exists
    if os.path.isfile('metadata.csv'):
        # If the file exists, read it
        return pd.read_csv('metadata.csv')
    else:
        try:
            # Get the metadata from the database
            # brut_metadata = get_metadata_from_mariadb_db(sql_database, sql_user, sql_password, sql_host)
            brut_metadata = get_metadata_from_postgres_db()
            # Clean the metadata
            cln_metadata = clean_metadata(brut_metadata)
            # Convert the metadata to a DataFrame
            df_metadata = pd.DataFrame.from_dict(cln_metadata).transpose()
            # Fill the 'Make' property NaN values with 'Undefined'
            df_metadata['Make'].fillna('Undefined', inplace=True)
            df_metadata.to_csv('metadata.csv', index=False, mode='w')
            return df_metadata

        except Exception as e:
            # remove the metadata file if an error occured to get the metadata from the database again
            if os.path.isfile('metadata.csv'):
                os.remove('metadata.csv')
            return Response(
                "Error while getting the metadata from the database, please retry",
                status=500,
                mimetype='application/json'
            )


@app.route('/reset', methods=['GET'])
def reset_metadata():
    """
    Reset the metadata file
    :return: A JSON object with the metadata
    """
    # remove the metadata file
    if os.path.isfile('metadata.csv'):
        os.remove('metadata.csv')

    return get_metadata()


def fig_to_buffer(fig):
    """
    Convert a figure to a buffer

    :param fig: The figure to convert
    :return: The buffer
    """
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)

    return buf


def merge_buffers_to_img(*buffers, max_columns=2):
    """
    Merge the images from the buffers into a single image.
    The images will be merged in a grid with a maximum number of columns.
    The images will be resized to fit the grid.

    :param buffers: The buffers containing the images to merge
    :param max_columns: The maximum number of columns
    :return: The buffer containing the merged image
    """
    # Define some constants
    IMAGE_FORMAT = 'RGB'
    MARGIN = 10

    # Open all the images from the buffers
    images = [Image.open(buffer) for buffer in buffers]

    # Calculate the new size for the merged image
    max_height = max(image.height for image in images)
    columns = min(len(images), max_columns)
    rows = (len(images) + columns - 1) // columns
    new_width = columns * (images[0].width + MARGIN) - MARGIN
    new_height = rows * (max_height + MARGIN) - MARGIN

    # Create a new blank image with the new size
    merged_image = Image.new(IMAGE_FORMAT, (new_width, new_height), color='white')

    # Paste the original images onto the new image
    x, y = 0, 0
    for image in images:
        merged_image.paste(image, (x, y))
        x += image.width + MARGIN
        if x >= columns * (images[0].width + MARGIN):
            x = 0
            y += max_height + MARGIN

    # Save the merged image to a new buffer
    merged_buffer = io.BytesIO()
    merged_image.save(merged_buffer, format='PNG')
    merged_buffer.seek(0)

    return merged_buffer


def display_bar(title, x_label, y_label, x_values, y_values, colors=None, rotation=90):
    """
    Display a bar chart

    :param title: The title of the chart
    :param x_label: The x-axis label
    :param y_label: The y-axis label
    :param x_values: The values of the x-axis
    :param y_values: The values of the y-axis
    :param colors: The colors of the bars
    :param rotation: The rotation of the x-axis labels
    """

    fig, ax = plt.subplots()
    ax.bar(x_values, y_values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(x_values, rotation=rotation)

    return fig_to_buffer(fig)


def display_pie(title, values, labels, colors=None, autopct="%1.1f%%", legend_title=None, legend_loc=None,
                legend_margin=None):
    """
    Display a pie chart

    :param title: The title of the chart
    :param values: The values of the chart
    :param labels: The labels of the chart
    :param colors: The colors of the chart
    :param autopct: The percentage format
    :param legend_title: The title of the legend,
    :param legend_loc: The location of the legend
    :param legend_margin: The margin of the legend
    """
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct=autopct, colors=colors)
    if legend_title is not None or legend_loc is not None or legend_margin is not None:
        ax.legend(title=legend_title, loc=legend_loc, bbox_to_anchor=legend_margin)
    ax.set_title(title)

    return fig_to_buffer(fig)


def display_curve(title, x_label, y_label, x_values, y_values, rotation=90):
    """
    Display a curve

    :param title: The title of the curve
    :param x_label: The label of the x_axis
    :param y_label: The label of the y_axis
    :param x_values: The values of the x_axis
    :param y_values: The values of the y_axis
    :param rotation: The rotation of the x_axis labels
    """

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_xticklabels(x_values, rotation=rotation)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    return fig_to_buffer(fig)


def display_histogram(title, x_label, y_label, x_values, bins=10, rotation=90):
    """
    Display a histogram

    :param title: The title of the histogram
    :param x_label: The label of the x_axis
    :param y_label: The label of the y_axis
    :param x_values: The values of the x_axis
    :param bins: The number of bins
    :param rotation: The rotation of the x_axis labels
    """

    fig, ax = plt.subplots()
    ax.hist(x_values, bins=bins)
    ax.set_xticklabels(x_values, rotation=rotation)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    return fig_to_buffer(fig)


def display_tree_map(title, sizes, labels, colors, alpha=0.6):
    """
    Display a tree map

    :param title: The title of the tree map
    :param sizes: The sizes of the tree map
    :param labels: The labels of the tree map
    :param colors: The colors of the tree map
    :param alpha: The alpha of the tree map
    """
    fig, ax = plt.subplots()
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=alpha, ax=ax)
    ax.set_title(title)

    return fig_to_buffer(fig)


def word_color_func(word, *args, **kwargs):
    """
    Get the corresponding color for a given word

    :param word: color name to find the corresponding color
    :return: the corresponding color in hex format
    """
    try:
        color_hex = webcolors.name_to_hex(word)
        return color_hex
    except ValueError:
        return 'black'


def display_wordcloud(words, frequencies, background_color='white', max_words=200, word_to_color=False):
    """
    Display a word cloud

    :param words: words to display
    :param frequencies: frequencies of the words
    :param background_color: background color of the word cloud
    :param max_words: maximum number of words to display
    :param word_to_color: if True, the words will be colored according to their name
    :return: the word cloud as a buffer
    """

    # Set the color function to convert the words to colors if needed
    if word_to_color:
        color_func = word_color_func
    else:
        color_func = None

    # Generate the word cloud
    wordcloud = WordCloud(
        background_color=background_color,
        max_words=max_words,
        color_func=color_func,  # Add the color_func parameter
    ).generate_from_frequencies(dict(zip(words, frequencies)))

    # Save the word cloud to a buffer
    buffer = io.BytesIO()
    wordcloud.to_image().save(buffer, 'PNG')
    buffer.seek(0)
    return buffer


def interval_check_to_int(nb_intervals):
    """
    Check if the interval is a valid integer

    :param nb_intervals: The interval check to check
    :return: The interval check as an integer
    """
    try:
        nb_intervals = int(nb_intervals)
        if nb_intervals < 1:
            raise ValueError
    except ValueError:
        return Response('Invalid interval', 400)

    return nb_intervals


def graph_type_check(graph_type, graph_types=None):
    """
    Check if the graph type is a valid string

    :param graph_type: The graph type to check
    :param graph_types: The list of valid graph types
    :return: The graph type as a string
    """
    if graph_types is None:
        graph_types = ['all', 'bar', 'pie', 'curve', 'histogram', 'tree_map', 'wordcloud']

    try:
        graph_type = str(graph_type)
        if graph_type not in graph_types:
            raise ValueError
    except ValueError:
        return Response('Invalid graph type', 400)

    return graph_type


@app.route('/graph/size/static', methods=['GET'])
@app.route('/graph/size/static/<interval_size>/<nb_intervals>', methods=['GET'])
def graph_images_size_static(interval_size=3000, nb_intervals=2):
    """
    Graph the number of images per size category

    :param interval_size: The size of the intervals
    :param nb_intervals: The number of intervals
    """
    # Get the metadata
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    # check values
    nb_intervals = interval_check_to_int(nb_intervals)

    # Convert 'ImageHeight' and 'ImageWidth' to numeric values
    df_meta['ImageHeight'] = pd.to_numeric(df_meta['ImageHeight'], errors='coerce')
    df_meta['ImageWidth'] = pd.to_numeric(df_meta['ImageWidth'], errors='coerce')

    # Drop rows with missing values
    df_meta = df_meta.dropna(subset=['ImageHeight', 'ImageWidth'])

    try:
        interval_size = int(interval_size)
    except ValueError:
        return 'Invalid interval size', 400

    # Calculate the minimum size of each image and store it in a new column
    df_meta['min_size'] = df_meta[['ImageWidth', 'ImageHeight']].min(axis=1)

    # Determine the maximum minimum size
    max_min_size = df_meta['min_size'].max()

    # Create a list of intervals based on the interval size and number of intervals
    inter = [i * interval_size for i in range(nb_intervals + 1)]

    # Create a list of labels for each interval
    labels = [f'{inter[i]}-{inter[i + 1]}' for i in range(nb_intervals)]

    # Categorize each image based on its size and interval
    df_meta['size_category'] = pd.cut(df_meta['min_size'], bins=inter, labels=labels)

    # Count the number of images in each category
    size_counts = df_meta['size_category'].value_counts()

    buffer = display_bar(title='Number of images per size category', x_label='Size category',
                         y_label='Number of images',
                         x_values=size_counts.index, y_values=size_counts.values)

    return Response(buffer.getvalue(), mimetype='image/png')


@app.route('/graph/size', methods=['GET'])
@app.route('/graph/size/dynamic', methods=['GET'])
@app.route('/graph/size/dynamic/<nb_intervals>/<graph_type>', methods=['GET'])
def graph_images_size_dynamic(nb_intervals=7, graph_type='all'):
    """
    Graph the number of images per size category
    The interval size is calculated dynamically

    :param nb_intervals: The number of intervals in the graph
    :param graph_type: The type of graph to display (bar, pie or all for both)
    """
    # Get the metadata
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    # check values
    nb_intervals = interval_check_to_int(nb_intervals)
    graph_type = graph_type_check(graph_type)

    # Convert 'ImageHeight' and 'ImageWidth' to numeric values
    df_meta['ImageHeight'] = pd.to_numeric(df_meta['ImageHeight'], errors='coerce')
    df_meta['ImageWidth'] = pd.to_numeric(df_meta['ImageWidth'], errors='coerce')

    # Drop rows with missing values
    df_meta = df_meta.dropna(subset=['ImageHeight', 'ImageWidth'])

    # Calculate the minimum size of each image and store it in a new column
    df_meta['min_size'] = df_meta[['ImageHeight', 'ImageWidth']].min(axis=1)

    # Determine the maximum minimum size and calculate the number of bins dynamically based on the number of columns
    max_min_size = df_meta['min_size'].max()
    num_images = len(df_meta)
    num_bins = int(num_images / (num_images / nb_intervals))

    # Create a list of bins based on the maximum minimum size and number of bins
    bins = [i * (max_min_size / num_bins) for i in range(num_bins + 1)]

    # Create a list of labels for each bin
    labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(num_bins)]

    # Categorize each image based on its size and bin
    df_meta['size_category'] = pd.cut(df_meta['min_size'], bins=bins, labels=labels)

    # Count the number of images in each category
    size_counts = df_meta['size_category'].value_counts()

    title = 'Number of images per size category'
    x_label = 'Image size'
    y_label = 'Number of images'

    # Create the appropriate chart based on the graph type parameter
    if graph_type == 'bar':
        buffer = display_bar(title=title, x_label=x_label, y_label=y_label,
                             x_values=size_counts.index, y_values=size_counts.values)
        return Response(buffer.getvalue(), mimetype='image/png')

    elif graph_type == 'pie':
        buffer = display_pie(title=title, values=size_counts.values, labels=size_counts.index)
        return Response(buffer.getvalue(), mimetype='image/png')

    elif graph_type == 'all':
        bar_buffer = display_bar(title=title, x_label=x_label, y_label=y_label,
                                 x_values=size_counts.index, y_values=size_counts.values)
        pie_buffer = display_pie(title=title, values=size_counts.values, labels=size_counts.index)
        merged_buffer = merge_buffers_to_img(bar_buffer, pie_buffer)

        return Response(merged_buffer.getvalue(), mimetype='image/png')

    else:
        return Response("Invalid graph type", 400)


def convert_to_year(date_str):
    """
    Convert a date string to a year

    :param date_str: The date string to convert
    :return: The year as an integer
    """
    try:
        dt = pd.to_datetime(date_str)
        if dt.year >= 1678:  # To avoid OutOfBoundsDatetime error
            return dt.year
        else:
            return None
    except:
        return None


@app.route('/graph/year', methods=['GET'])
@app.route('/graph/year/<nb_intervals>/<graph_type>', methods=['GET'])
def graph_images_year(nb_intervals=10, graph_type='all'):
    """
    Graph the number of images per year

    :param graph_type: The type of graph to display (bar, pie, curve or all for all)
    :param nb_intervals: The number of intervals to display
    """
    # Get the metadata
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    # Check the values
    graph_type = graph_type_check(graph_type)
    nb_intervals = interval_check_to_int(nb_intervals)

    # Extract year from the 'DateTime' column and create a new 'Year' column
    df_meta['Year'] = df_meta['DateTimeOriginal'].apply(convert_to_year)

    # Remove rows with invalid years (None)
    df_meta = df_meta.dropna(subset=['Year'])

    # Group the data by year and count the number of images for each year
    image_count = df_meta.groupby('Year').size().reset_index(name='count')[:nb_intervals]
    image_count['Year'] = image_count['Year'].astype(int)

    # Set the title of the graph
    title = 'Number of images per year'
    x_label = 'Year'
    y_label = 'Number of images'

    # Display different types of graphs based on the 'graph_type' parameter
    if graph_type == 'bar':
        # Display a bar chart
        image_count.plot(kind='bar', x='Year', y='count')
        buffer = display_bar(title=title, x_label=x_label, y_label=y_label, x_values=image_count['Year'],
                             y_values=image_count['count'])
        return Response(buffer.getvalue(), mimetype='image/png')

    elif graph_type == 'pie':
        # Display a pie chart using a custom function 'display_pie'
        buffer = display_pie(title=title, values=image_count['count'], labels=image_count['Year'])
        return Response(buffer.getvalue(), mimetype='image/png')

    elif graph_type == 'curve':
        # Display a line chart using a custom function 'display_curve'
        image_count = df_meta.groupby('Year').size().reset_index(name='count')
        buffer = display_curve(title=title, x_label=x_label, y_label=y_label, x_values=image_count['Year'],
                               y_values=image_count['count'])
        return Response(buffer.getvalue(), mimetype='image/png')

    elif graph_type == 'wordcloud':
        # Display a word cloud
        buffer = display_wordcloud(words=list(image_count['Year'].astype(str)), frequencies=list(image_count['count']))
        return Response(buffer.getvalue(), mimetype='image/png')

    elif graph_type == 'all':
        # Display all three types of graphs: bar, pie, and line charts

        # Bar chart
        image_count.plot(kind='bar', x='Year', y='count')
        buffer_bar = display_bar(title=title, x_label=x_label, y_label=y_label, x_values=image_count['Year'],
                                 y_values=image_count['count'])

        # Pie chart
        buffer_pie = display_pie(title=title, values=image_count['count'], labels=image_count['Year'])

        # Line chart
        buffer_line = display_curve(title=title, x_label=x_label, y_label=y_label, x_values=image_count['Year'],
                                    y_values=image_count['count'])

        # Word cloud
        buffer_wordcloud = display_wordcloud(words=list(image_count['Year'].astype(str)),
                                             frequencies=list(image_count['count']))

        # Merge the three graphs into one image
        merged_buffer = merge_buffers_to_img(buffer_bar, buffer_pie, buffer_line, buffer_wordcloud)

        return Response(merged_buffer.getvalue(), mimetype='image/png')
    else:
        # Raise an error if an invalid 'graph_type' parameter is passed
        return Response("Invalid graph type", 400)


@app.route('/graph/brand', methods=['GET'])
@app.route('/graph/brand/<nb_columns>/<graph_type>', methods=['GET'])
def graph_images_brand(graph_type='all', nb_columns=5):
    """
    Graph the number of images per brand

    :param graph_type: The type of graph to display (bar, pie or all for both)
    :param nb_columns: The number of columns to display
    """
    # Get the metadata
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    # Fill the missing values with 'Undefined'
    df_meta['Make'].fillna('Undefined', inplace=True)

    # Check the values
    graph_type = graph_type_check(graph_type)
    nb_columns = interval_check_to_int(nb_columns)

    # Initialize an empty dictionary to store the counts of each brand
    counts = {}

    # Loop through each brand in the metadata and count the number of occurrences
    for make in df_meta['Make']:
        if make is not None:
            counts[make] = counts.get(make, 0) + 1

    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    # Convert the dictionary into two lists of labels and values for graphing
    labels = list(sorted_counts.keys())[:nb_columns]
    values = list(sorted_counts.values())[:nb_columns]

    # Set the title for the graph
    title = 'Number of images per brand'
    x_label = 'Brand'
    y_label = 'Number of images'

    # Determine which type of graph to display based on the 'graph_type' parameter
    if graph_type == 'bar':
        # Display a bar graph
        buffer = display_bar(title=title, x_label=x_label, y_label=y_label, x_values=labels, y_values=values)
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph_type == 'pie':
        # Display a pie chart
        buffer = display_pie(title=title, values=values, labels=labels)
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph_type == 'wordcloud':
        # Display a word cloud
        buffer = display_wordcloud(words=labels, frequencies=values)
        return Response(buffer.getvalue(), mimetype='image/png')

    elif graph_type == 'all':
        # Display both a bar graph and a pie chart
        buffer_bar = display_bar(title=title, x_label=x_label, y_label=y_label, x_values=labels, y_values=values)
        buffer_pie = display_pie(title=title, values=values, labels=labels)
        buffer_wordcloud = display_wordcloud(words=labels, frequencies=values)

        # Merge the two graphs into one image
        merged_buffer = merge_buffers_to_img(buffer_bar, buffer_pie, buffer_wordcloud)
        return Response(merged_buffer.getvalue(), mimetype='image/png')

    else:
        # Raise an error if the 'graph_type' parameter is invalid
        return Response("Invalid graph type", 400)


def get_coordinates(df_meta, country=False):
    """
    Extract the coordinates of the images with GPS data

    :param df_meta: The metadata to extract the coordinates from
    :param country: Whether to get the country information or not
    """
    coords = {}
    for file, lattitude, longitude, altitude in zip(
            df_meta['filename'],
            df_meta['Latitude'],
            df_meta['Longitude'],
            df_meta['Altitude']
    ):
        if lattitude is not None and not np.isnan(lattitude) and lattitude != 0.0 \
                and longitude is not None and not np.isnan(longitude) and longitude != 0.0:
            coords.update({file: [lattitude, longitude, altitude]})

    if len(coords) == 0:
        return None

    if country:
        return get_country(coords)
    else:
        return coords


def get_country(coordinates):
    """
    Get the country of each coordinate

    :param coordinates: The coordinates to get the country from
    :return: The coordinates with the country added
    """

    # Create a geolocator
    geolocator = Nominatim(user_agent="geoapiExercises")
    coordinates_list = coordinates.copy()

    # Get the continent information for each coordinate
    for key, coord in tqdm(coordinates_list.items(), desc='Getting country information'):
        if len(coord) < 4:  # If the country hasn't been found yet
            try:
                location = geolocator.reverse(coord, exactly_one=True, language='en')
                address = location.raw['address']
                country = address.get('country')
                coordinates[key].append(country)
            except:
                print(f"Error with {key} : {coord}")

    return coordinates


@app.route('/graph/map', methods=['GET'])
def display_coordinates_on_map():
    """
    Display the coordinates on a map
    """

    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    coordinates_list = get_coordinates(df_meta, False)

    if coordinates_list is None:
        return Response("No coordinates found", 400)

    # create a map centered at a specific location
    m = folium.Map(location=[0, 0], zoom_start=1)

    # add markers for each set of coordinates
    for image, coords in coordinates_list.items():
        lat, lon, alt = coords
        folium.Marker(location=[lat, lon], tooltip=image, popup=f'file:{image}\ncoord:{coords}').add_to(m)

    # Save map to HTML
    m.save('map.html')

    return send_file('map.html', mimetype='text/html')


@app.route('/graph/countries', methods=['GET'])
@app.route('/graph/countries/<int:nb_inter>/<graph>', methods=['GET'])
def graph_images_countries(nb_inter=5, graph='all'):
    """
    Display graphs about the number of images by country

    :param nb_inter: number of countries to display
    :param graph: type of graph to display (bar, pie, all)
    """
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    coord_list = get_coordinates(df_meta, True)

    if coord_list is None:
        return Response("No coordinates found", 400)

    graph = graph_type_check(graph)
    nb_inter = interval_check_to_int(nb_inter)

    # Create a pandas DataFrame from the coordinates dictionary
    df = pd.DataFrame.from_dict(coord_list, orient='index',
                                columns=['Latitude', 'Longitude', 'Altitude', 'Country'])

    # Group the DataFrame by continent and count the number of images
    country_count = df.groupby('Country')['Country'].count()
    country_count = country_count.sort_values(ascending=False)[:nb_inter]

    title = 'Number of images by country'
    x_label = 'Country'
    y_label = 'Image Count'

    if graph == 'bar':
        buffer = display_bar(title=title, x_label=x_label, y_label=y_label,
                             x_values=country_count.index, y_values=country_count.values)
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'pie':
        buffer = display_pie(title=title, values=country_count.values, labels=country_count.index)
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'wordcloud':
        # Display a word cloud
        buffer = display_wordcloud(words=country_count.index, frequencies=country_count.values)
        return Response(buffer.getvalue(), mimetype='image/png')
    else:
        buffer_bar = display_bar(title=title, x_label=x_label, y_label=y_label,
                                 x_values=country_count.index, y_values=country_count.values)
        buffer_pie = display_pie(title=title, values=country_count.values, labels=country_count.index)
        buffer_wordcloud = display_wordcloud(words=country_count.index, frequencies=country_count.values)

        # Merge the two buffers into a single buffer
        combined_buffer = merge_buffers_to_img(buffer_bar, buffer_pie, buffer_wordcloud)
        return Response(combined_buffer.getvalue(), mimetype='image/png')


@app.route('/graph/altitude', methods=['GET'])
@app.route('/graph/altitude/<int:nb_inter>/<graph>', methods=['GET'])
def graph_images_altitudes(nb_inter=5, graph='all'):
    """
    Display graphs about the number of images by altitude.

    :param nb_inter: number of interval
    :param graph: type of graph to display (histogram, pie, all)
    """
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    coord_list = get_coordinates(df_meta, False)

    if coord_list is None:
        return Response("No coordinates found", 400)

    graph = graph_type_check(graph)
    nb_inter = interval_check_to_int(nb_inter)

    altitudes = []
    for img in coord_list:
        alt = float(coord_list[img][2])
        if alt > 0.0:
            altitudes.append(alt)

    # Créer les intervalles en utilisant linspace() de numpy
    intervalles = np.linspace(0, max(altitudes), nb_inter + 1)

    # Convertir les intervalles en paires d'intervalles
    intervalles = [(int(intervalles[i]), int(intervalles[i + 1])) for i in range(len(intervalles) - 1)]

    # Compte combien d'altitudes se situent dans chaque intervalle
    counts = [0] * len(intervalles)
    for altitude in altitudes:
        for i, intervalle in enumerate(intervalles):
            if intervalle[0] <= altitude < intervalle[1]:
                counts[i] += 1

    # Créer une liste de noms pour les intervalles
    noms_intervalles = ["{}-{}".format(intervalle[0], intervalle[1]) for intervalle in intervalles]

    title = 'Number of images by altitude'
    x_label = 'Altitude'
    y_label = 'Image Count'

    if graph == 'histogram':
        buffer = display_histogram(title=title, x_label=x_label, y_label=y_label, x_values=altitudes, bins=nb_inter)
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'pie':
        buffer = display_pie(title=title, values=counts, labels=noms_intervalles)
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'bar':
        buffer = display_bar(title=title, x_label=x_label, y_label=y_label, x_values=noms_intervalles, y_values=counts)
        return Response(buffer.getvalue(), mimetype='image/png')
    else:
        buffer_histo = display_histogram(title=title, x_label=x_label, y_label=y_label, x_values=altitudes,
                                         bins=nb_inter)
        buffer_bar = display_bar(title=title, x_label=x_label, y_label=y_label, x_values=noms_intervalles,
                                 y_values=counts)
        buffer_pie = display_pie(title=title, values=counts, labels=noms_intervalles)

        # Merge the three buffers into a single one
        combined_buffer = merge_buffers_to_img(buffer_histo, buffer_bar, buffer_pie)
        return Response(combined_buffer.getvalue(), mimetype='image/png')


def closest_colour(requested_colour):
    """
    Find the closest color in the webcolors library

    :param requested_colour: color to find
    :return: the closest color
    """
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    """
    Get the name of the closest color

    :param requested_colour: color to find
    :return: the actual name and the closest name
    """
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


@app.route('/graph/dominant_color', methods=['GET'])
@app.route('/graph/dominant_color/<nb_inter>/<graph>', methods=['GET'])
def graph_dominant_colors(nb_inter=20, graph='all'):
    """
    Display graphs about the number of images by dominant color

    :param nb_inter: number of colors
    :param graph: type of graph to display (bar, pie, treemap, all)
    """
    # Get the metadata
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    # Check the parameters
    graph = graph_type_check(graph)
    nb_inter = interval_check_to_int(nb_inter)

    # Create a dictionary of dominant colors
    dict_dom_color = {}
    df_dict_meta = df_meta["dominant_color"].to_dict()

    # convert string of dom color to list
    for img in df_dict_meta:
        try:
            if df_dict_meta[img] is not None and df_dict_meta[img] is not np.nan:
                list_dom_color = eval(df_dict_meta[img])
                dict_dom_color.update({img: list_dom_color})
        except:
            print(f"Error with {img} : {df_dict_meta[img]}")

    # Count the number of times each color appears
    color_counts = Counter()
    for image_colors in dict_dom_color.values():
        for color, percentage in image_colors:
            color_counts[color] += percentage

    # Map hexadecimal codes to color names
    color_names = {}
    for code in color_counts.keys():
        try:
            rgb = webcolors.hex_to_rgb(code)
            actual, closest = get_colour_name(rgb)
            color_names[code] = closest
        except ValueError:
            pass

    # Create a dictionary of color percentages
    dict_res = {}
    for key, val in color_names.items():
        if val in dict_res:
            dict_res[val] += round(color_counts[key] / 100, 5)
        else:
            dict_res[val] = round(color_counts[key] / 100, 5)

    # Create a bar graph showing the dominant colors in the images
    if sum(dict_res.values()) > 100:
        raise Exception('Error : sum of percentages is greater than 100')

    columns = dict_res.__len__()
    if columns > nb_inter: columns = nb_inter

    # Sort the dictionary by value
    sorted_colors = sorted(dict_res.items(), key=lambda x: x[1], reverse=True)
    top_colors = dict(sorted_colors[:columns])
    color_labels = list(top_colors.keys())
    sizes = list(top_colors.values())
    color = [webcolors.name_to_hex(c) for c in top_colors]

    title = 'Top Colors'
    x_label = 'Color'
    y_label = 'Percentage'

    if graph == 'bar':
        buffer = display_bar(title=title, x_label=x_label, y_label=y_label, colors=top_colors.keys(),
                             x_values=top_colors.keys(), y_values=top_colors.values())
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'pie':
        buffer = display_pie(title=title, values=top_colors.values(), labels=top_colors.keys(), colors=color_labels)
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'treemap':
        buffer = display_tree_map(title=title, sizes=sizes, labels=color_labels, colors=color, alpha=.7)
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'wordcloud':
        buffer = display_wordcloud(words=color_labels, frequencies=sizes, word_to_color=True)
        return Response(buffer.getvalue(), mimetype='image/png')
    else:
        buffer_bar = display_bar(title=title, x_label=x_label, y_label=y_label, colors=top_colors.keys(),
                                 x_values=top_colors.keys(), y_values=top_colors.values())
        buffer_pie = display_pie(title=title, values=top_colors.values(), labels=top_colors.keys(), colors=color_labels)
        buffer_treemap = display_tree_map(title=title, sizes=sizes, labels=color_labels, colors=color, alpha=.7)
        buffer_wordcloud = display_wordcloud(words=color_labels, frequencies=sizes, word_to_color=True)

        # combine the 3 graphs
        combined_buffer = merge_buffers_to_img(buffer_bar, buffer_pie, buffer_treemap, buffer_wordcloud)
        return Response(combined_buffer.getvalue(), mimetype='image/png')


@app.route('/graph/tags/top', methods=['GET'])
@app.route('/graph/tags/top/<nb_inter>/<graph>', methods=['GET'])
def graph_top_tags(nb_inter=5, graph='all'):
    """
    Display graphs about the number of images by tags

    :param nb_inter: number of tags
    :param graph: type of graph to display (bar, pie, all)
    """
    # get the metadata
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    # Check the parameters
    graph = graph_type_check(graph)

    nb_inter = interval_check_to_int(nb_inter)

    # get top nb_inter tags
    tags = df_meta['tags'].dropna()  # Remove NaN values
    # Convert string representation of lists to actual lists
    tags = tags.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    # Flatten the list of lists and remove empty strings
    all_tags = pd.Series([tag for sublist in tags for tag in sublist if tag != '[]'])
    # Get the most frequent tags
    top_tags = all_tags.value_counts().nlargest(nb_inter).to_dict()

    title = 'Top Tags'
    x_label = 'Tag'
    y_label = 'Count'

    if graph == 'bar':
        buffer = display_bar(title=title, x_label=x_label, y_label=y_label,
                             x_values=top_tags.keys(), y_values=top_tags.values())
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'pie':
        buffer = display_pie(title=title, values=top_tags.values(), labels=top_tags.keys())
        return Response(buffer.getvalue(), mimetype='image/png')
    elif graph == 'wordcloud':
        # Display a word cloud
        buffer = display_wordcloud(words=list(top_tags.keys()), frequencies=list(top_tags.values()))
        return Response(buffer.getvalue(), mimetype='image/png')

    else:
        buffer_bar = display_bar(title=title, x_label=x_label, y_label=y_label,
                                 x_values=top_tags.keys(), y_values=top_tags.values())
        buffer_pie = display_pie(title=title, values=top_tags.values(), labels=top_tags.keys())
        buffer_wordcloud = display_wordcloud(words=list(top_tags.keys()), frequencies=list(top_tags.values()))

        # combine the 2 graphs
        combined_buffer = merge_buffers_to_img(buffer_bar, buffer_pie, buffer_wordcloud)
        return Response(combined_buffer.getvalue(), mimetype='image/png')


def categorize_tags(df_meta, categories_list: list):
    """
    Categorize tags based on similarity to category prototypes

    :param categories_list: list of categories
    :param df_meta: DataFrame of metadata
    :return: dictionary of categories
    """
    # Concatenate all tags in a list
    all_tags = []
    for tags in df_meta['tags']:
        try:
            if isinstance(tags, str):
                tags = eval(tags)
            if tags is not None and tags is not np.nan:
                all_tags += tags
        except:
            print("Error : ", tags)

    # Load pre-trained word embedding model
    nlp = spacy.load("en_core_web_md")

    categories = {}
    for cate in categories_list:
        categories[cate] = {}

    # categorize words based on similarity to category prototypes
    for word in tqdm(all_tags, desc="Categorizing tags"):
        # find the most similar category prototype for the word
        max_similarity = -1
        chosen_category = "other"
        for category in categories:
            similarity = nlp(word).similarity(nlp(category))
            if similarity > max_similarity:
                max_similarity = similarity
                chosen_category = category

        # add the word into the appropriate category dictionary
        categories[chosen_category].update({word: max_similarity})

    return categories


@app.route('/graph/tags/dendrogram', methods=['GET'])
def graph_categorized_tags():
    """
    Display a Dendrogram of categorized tags
    """
    # get the metadata
    df_meta = get_metadata()
    if isinstance(df_meta, Response):
        return df_meta

    # Check if the request body contains JSON data
    if request.is_json:
        # Get the list of categories from the request
        categories_list = request.get_json().get('list')
    else:
        categories_list = None

    # Verify if the categories_list is a list and not empty
    if categories_list is None or not isinstance(categories_list, list) or len(categories_list) == 0:
        print("No list of categories provided, using default list")
        # default list of categories
        categories_list = [
            'Fruit', 'Animal', 'Electronics', 'Furniture', 'Vehicle',
            'Clothing', 'Sport', 'Kitchen', 'Outdoor', 'Accessory'
        ]

    categorized_tags = categorize_tags(df_meta, categories_list)

    keys_and_subkeys = []
    for key, subdict in categorized_tags.items():
        for subkey in subdict:
            keys_and_subkeys.append((key, subkey))

    labels = [f"{key} -> {subkey}" for key, subkey in keys_and_subkeys]

    def custom_distance(x, y):
        key1, subkey1 = x
        key2, subkey2 = y
        if key1 == key2:
            return abs(categorized_tags[key1][subkey1] - categorized_tags[key2][subkey2])
        else:
            return 1.0

    dist_matrix = pdist(keys_and_subkeys, custom_distance)
    Z = linkage(dist_matrix, method='average')

    fig = plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=labels, orientation='top', leaf_font_size=10)
    plt.xlabel("Distance")
    plt.tight_layout()

    # Save the plot to a buffer
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)

    # Return the buffer contents as a response
    return Response(buffer.getvalue(), mimetype='image/png')


@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True)
