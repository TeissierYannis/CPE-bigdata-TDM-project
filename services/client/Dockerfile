# Use the official Node.js image as a base
FROM node:16-alpine

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
EXPOSE 4005

# Start the application
CMD ["npm", "start"]
