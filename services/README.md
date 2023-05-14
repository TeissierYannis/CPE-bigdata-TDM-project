# Services

# Sources : 

> https://github.com/TeissierYannis/CPE-bigdata-TDM-project

## Installation

To install the services, you need to have docker and docker-compose installed on your machine.
The simplest way is to install docker desktop.

Then, run the following command to start the services :

```bash
docker-compose up -d --build
```

Wait for all services to be up and running. (30seconds to 1 minute)

```bash
curl 127.0.0.1:81/gather/download
```

Follow the progress of the download :

```bash
curl 127.0.0.1:81/gather/status
```

Now, look at the harvest_worker_services logs to see the progress of the harvest.

```bash
docker-compose logs -f harvest_worker_service
```

### Front-end

To access the front end, on your browser, go to the following url :

```
http://127.0.0.1:4005
```

You will find an interface to test the recommend service.
Please do not use it before the download is finished.


### lab

To access the lab, on your browser, go to the following url :

```bash
http://127.0.0.1:8888/lab
```

In this lab, you will find a notebook to use the metadata that you can get from the database.
This is for experimentation purpose only.
Deactivate the lab if you want to use it in production.

## Endpoints

You will find a postman library to test each of the following endpoints.

They all have a '/api/v1/health' endpoint to check if the service is up and running.
So you can run the following command to check the status of the service :

```bash
curl http://127.0.0.1:81/<service>/api/v1/health
```

service list : gather, cdn, recommend or visualize

### Gather

As explain in the installation section, you can start the download a get request on the following endpoint :

```bash
curl http://127.0.0.1:81/gather/download
```

This will start the download of the data from the source and store it in the database.
This will automatically get the dataset, store the images, extract the metadata,
evaluate tags and dominant colors and store the data in the databases.

You can also get the status of the download with the following endpoint :

```bash
curl http://127.0.0.1:81/gather/status
```

### CDN

The CDN is a simple nginx server that serve the images from the database.

You can check the status of the CDN with the following endpoint :

```bash
curl http://127.0.0.1:81/cdn/api/v1/health
```

The following endpoint will return the image with the given name :

```bash
curl http://127.0.0.1:81/cdn/show/<filename>
```

### Recommend

The recommend service is available at the following endpoint :

```bash
curl --location 'http://127.0.0.1:81/recommend/recommend' \
  --header 'Content-Type: application/json' \
  --data '{
    "preferences": {
        "dominant_color": "#b14343",
        "imagewidth": 800,
        "imageheight": 600,
        "orientation": 0,
        "tags": [
            "cat",
            "person"
        ],
        "make": "Canon"
    }
}'
```

This will return a list of images id that match the preferences.

### Visualise

The visualise service is available through many endpoints :

- **metadata reset**:

Since it is possible to use the metadata in several database reports, it is necessary to be able to obtain the latest version available. This route allows you to retrieve all the metadata again.

```bash
curl http://127.0.0.1:81/visualize/reset
```

- **images size static**:

```bash
curl http://127.0.0.1:81/visualize/graph/size/static
```

or

```bash
curl http://127.0.0.1:81/visualize/graph/size/static/<interval_size>/<nb_intervals>
```

This will return a graph of the number of images per size interval.

- **images size dynamic**:

```bash
curl http://127.0.0.1:81/graph/size
```

or

```bash
curl http://127.0.0.1:81/graph/size/dynamic
```

or

```bash
curl http://127.0.0.1:81/graph/size/dynamic/<nb_intervals>/<graph_type>
```

This will return a graph of the number of images per size interval.
The intervals are dynamically calculated.
You can choose the number of intervals and the type of graph (bar, pie or all)

- **images year**:

```bash
curl http://127.0.0.1:81/graph/year
```

or

```bash
curl http://127.0.0.1:81/graph/year/<nb_intervals>/<graph_type>
```

This will return a graph of the number of images per year.
You can choose the number of intervals and the type of graph (bar, pie, curve, wordcloud or all)

- **Camera brand**:

```bash
curl http://127.0.0.1:81/graph/brand
```

or

```bash
curl http://127.0.0.1:81/graph/brand/<nb_columns>/<graph_type>
```

This will return a graph of the number of images per camera brand.
You can choose the number of columns and the type of graph (bar, pie, wordcloud or all)

- **Map of the images**:

```bash
curl http://127.0.0.1:81/map
```

This will return a html map of the images with the gps coordinates.
The simplest way to see the map is to use this request in a browser.

- **images countries**:

```bash
curl http://127.0.0.1:81/graph/countries
```

or

```bash
curl http://127.0.0.1:81/graph/countries/<nb_countries>/<graph_type>
```

This will return a graph of the number of images per country.
It will take a bit of time because it needs to get the country of each image using the gps coordinates.
You can choose the number of countries and the type of graph (bar, pie or all)

- **images altitude**:

```bash
curl http://127.0.0.1:81/graph/altitude
```

or

```bash
curl http://127.0.0.1:81/graph/altitude/<nb_intervals>/<graph_type>
```

This will return a graph of the number of images per altitude interval.
You can choose the number of intervals and the type of graph (histogram, pie or all)

- **images dominant color**:

```bash
curl http://127.0.0.1:81/graph/dominant_color
```

or

```bash
curl http://127.0.0.1:81/graph/dominant_color/<nb_colors>/<graph_type>
```

This will return a graph of the dominant color of the images.
You can choose the number of colors and the type of graph (bar, treemap, pie, wordcloud or all)

- **images tags**:

```bash
curl http://127.0.0.1:81/graph/tags/top
```

or

```bash
curl http://127.0.0.1:81/graph/tags/top/<nb_tags>/<graph_type>
```

This will return a graph of the most common object in the images.
You can choose the number of tags and the type of graph (bar, pie, wordcloud or all)

- **images dendrogram of tags**:

```bash
curl http://127.0.0.1:81/graph/tags/dendrogram
```

This will return a dendrogram of the tags of the images.
It will take a bit of time because it needs to classify the tags of each image.
you can put a list of categories in the body of the request if you want to use your own categories :

```json
{
  "list": [
    "Fruit",
    "Animal",
    "Electronics",
    "Furniture",
    "Vehicle",
    "Clothing",
    "Sport",
    "Kitchen",
    "Outdoor",
    "Accessory"
  ]
}
```

The default categories are the ones above.










