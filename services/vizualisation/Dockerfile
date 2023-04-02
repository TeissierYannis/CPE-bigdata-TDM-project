# Use an official Python runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app


# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install en_core_web_lg model for spaCy
RUN python -m spacy download en_core_web_lg

# Copy the rest of the application code into the container
COPY . .

# Expose port 4003 for the Flask app to listen on
EXPOSE 4003

# Set the environment variable for Flask
ENV FLASK_APP=app.py

RUN pip install timm

# Start the Flask app when the container launches
CMD sleep 10 && flask run --host=0.0.0.0 --port=4003