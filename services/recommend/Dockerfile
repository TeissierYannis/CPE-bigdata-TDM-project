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

# Expose port 80 for the Flask app to listen on
EXPOSE 4004

# Set the environment variable for Flask
ENV FLASK_APP=app.py

# Start the Flask app when the container launches
CMD sleep 10 && flask run --host=0.0.0.0 --port=4004
