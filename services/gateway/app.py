import logging

import requests
import os

from flask import Flask, jsonify, request
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')


# Define a sample endpoint
@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/<service>/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def redirect(service, path):
    # Print the parameters for debugging
    if service == 'gather':
        url = os.getenv('GATHER_SERVICE_URL') + path
    elif service == 'harvest':
        url = os.getenv('HARVEST_SERVICE_URL') + path
    elif service == "recommend":
        url = os.getenv('RECOMMEND_SERVICE_URL') + path
    elif service == "visualize":
        url = os.getenv('VISUALIZE_SERVICE_URL') + path
    elif service == "cdn":
        url = os.getenv('CDN_SERVICE_URL') + path
    else:
        return jsonify({'status': 'error', 'message': 'service not found'})

    response = requests.request(
        method=request.method,
        url=url,
        headers={key: value for (key, value) in request.headers if key != 'Host'},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False
    )

    return response.content, response.status_code, response.headers.items()


if __name__ == '__main__':
    app.run(debug=True)
