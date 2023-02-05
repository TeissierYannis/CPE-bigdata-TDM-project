import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to a specific address and port
server_address = ('localhost', 12345)
print('Starting up on {} port {}'.format(*server_address))
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)


def process_request(request):
    # Your code for processing the request here
    if request == 'download':
        return "download"
    elif request == 'get-metadata':
        return "get-metadata"
    elif request == 'get-colors':
        return "get-colors"


while True:
    # Wait for a connection
    print('Waiting for a connection')
    connection, client_address = sock.accept()
    try:
        print('Connection from', client_address)
        # Receive the data in small chunks and retransmit it
        while True:
            data = connection.recv(1024)
            if data:
                try:
                    request = data.decode().strip()
                    # Check if the request is a valid command
                    if request in ('download', 'get-metadata', 'get-colors'):
                        response = process_request(request)
                        connection.sendall(response.encode())
                    else:
                        connection.sendall('Error: Invalid Command'.encode())
                except Exception as e:
                    print('Error:', str(e))
                    connection.sendall('Error: Server Error'.encode())
            else:
                break
    except socket.error as e:
        print('Error:', str(e))
    finally:
        # Clean up the connection
        connection.close()
