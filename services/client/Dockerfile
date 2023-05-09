# Use the official Node.js image as a base
FROM node:16-alpine

# Install necessary packages for building certain Python libraries
RUN apk add --no-cache python3 py3-pip gcc musl-dev python3-dev linux-headers

# Install Rust and Cargo
RUN apk add --no-cache rust cargo

# Install tini for proper signal handling
RUN apk add --no-cache tini

# Install Jupyter Notebook, JupyterLab, and other dependencies
RUN pip3 install --upgrade pip
RUN pip3 install notebook jupyterlab

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json for installing dependencies
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy the application source code
COPY . .

# Build the application
RUN npm run build

# Expose the port the app runs on and the JupyterLab port
EXPOSE 4005 8888

# Start the application and JupyterLab server
CMD ["tini", "--", "sh", "-c", "jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --LabApp.token='' & npm start"]
