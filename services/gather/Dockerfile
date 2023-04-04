# Use an official Python runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create a folder config
RUN mkdir config

# Download the file from https://unsplash.com/data/lite/latest
RUN wget https://unsplash.com/data/lite/latest

# Extract the zip file
RUN unzip latest

# Copy the file to the photos.tsv000 to the config folder
RUN cp photos.tsv000 config/photos.tsv000

# Remove the zip file and all the other files
RUN rm latest

# Expose port 80 for the Flask app to listen on
EXPOSE 4001

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Start the Flask app when the container launches
CMD sleep 10 && flask run --host=0.0.0.0 --port=4001
