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

    # Before running, make sure all services are up and running
    # Send an http request to each service to check if it is up and running on the URL URL + /api/v1/health
    # If the service is not up and running, return an error message

    URLs = [
        os.getenv('GATHER_SERVICE_URL') + '/api/v1/health',
        os.getenv('HARVEST_SERVICE_URL') + '/api/v1/health',
        os.getenv('RECOMMEND_SERVICE_URL') + '/api/v1/health',
        os.getenv('VISUALIZE_SERVICE_URL') + '/api/v1/health',
        os.getenv('CDN_SERVICE_URL') + '/api/v1/health'
    ]

    for URL in URLs:
        try:
            response = requests.request(
                method='GET',
                url=URL,
                headers={key: value for (key, value) in request.headers if key != 'Host'},
                data=request.get_data(),
                cookies=request.cookies,
                allow_redirects=False
            )
        except Exception as e:
            logging.error(e)
            return jsonify({'status': 'error', 'message': 'service not found'})

        if response.status_code != 200:
            return jsonify({'status': 'error', 'message': 'service not running, please try again later'})


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
