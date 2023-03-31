# Use an official Python runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app

# Install the required packages for cv2
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 80 for the Flask app to listen on
EXPOSE 4002

# Set the environment variable for Flask
ENV FLASK_APP=app.py

RUN pip install timm

# Start the Flask app when the container launches
CMD sleep 10 && flask run --host=0.0.0.0 --port=4002