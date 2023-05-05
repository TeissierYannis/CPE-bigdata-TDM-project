# Use the official Node.js image as a base
FROM node:16-alpine

# Install necessary packages for building certain Python libraries
RUN apk add --no-cache python3 py-pip gcc musl-dev python3-dev linux-headers

# Install Rust and Cargo
RUN apk add --no-cache rust cargo

# Install tini for proper signal handling
RUN apk add --no-cache tini

# Install Jupyter Notebook, JupyterLab, and other dependencies
RUN pip install --upgrade pip
RUN pip install notebook jupyterlab

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json for installing dependencies
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the application source code
COPY . .

# Copy the JupyterLab configuration file
COPY jupyter_notebook_config.py /root/.jupyter/

# Build the application
RUN npm run build

# Expose the port the app runs on and the JupyterLab port
EXPOSE 4005 8888

# Start the application and JupyterLab server
CMD ["tini", "--", "sh", "-c", "jupyter lab --config=/root/.jupyter/jupyter_notebook_config.py --port=8888 & npm start"]
