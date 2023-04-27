FROM python

# Set the working directory to /app
WORKDIR /app

COPY . /

RUN apt-get update && apt-get install -y cmake

RUN git clone https://github.com/TeissierYannis/cpe-bigdata-project-cpp-dependencies.git .

RUN mkdir build

WORKDIR /app/build

RUN cmake ..

RUN cmake --build .

WORKDIR /app

# Move the exifextract binary to the shared folder
RUN mkdir -p /app/shared/config

RUN cp /app/build/exifextract /app/shared/exifextract

RUN chmod +x /export_to_shared

RUN cp /export_to_shared /app/export_to_shared

# Launch sleep 20s to wait for the other services to be ready and close the container
CMD sleep 60 && /app/export_to_shared