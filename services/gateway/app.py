from flask import Flask, jsonify, request

app = Flask(__name__)

# Define a sample endpoint
@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello, World!'})

# Define a sample POST endpoint
@app.route('/greet', methods=['POST'])
def greet():
    data = request.get_json()
    name = data['name']
    return jsonify({'message': f'Hello, {name}!'})

if __name__ == '__main__':
    app.run(debug=True)