import logging
from flask import Flask, jsonify
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Initialize the MinIO client
minio_client = Minio(
    "minio:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)

bucket_name = 'images'


# Define a sample endpoint
@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/show/<filename>', methods=['GET'])
def get_file(filename):
    # Print the parameters for debugging
    logging.info(f"Getting file {filename}")
    try:
        file = minio_client.get_object(bucket_name, filename)
        return file.data, 200, {'Content-Type': 'image/jpeg'}
    except Exception as e:
        logging.error(e)
        return jsonify({'status': 'error', 'message': 'file not found'})


if __name__ == '__main__':
    app.run(debug=True)
