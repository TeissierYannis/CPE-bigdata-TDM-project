# Setup

````bash
docker-compose build --no-cache
docker-compose up -d --build
````

# Curl Requests in the order they are called

## Gather services :

### 1. Gather the data from the source

```bash
curl -X GET http://127.0.0.1:80/gather/download
```

### 2. Check the status of the download

```bash
curl -X GET http://127.0.0.1:80/gather/status
```

### 3. Cancel the download

```bash
curl -X GET http://127.0.0.1:80/gather/cancel
```

## Harvest service :

### 1. Harvest the metadata

```bash
curl -X GET http://127.0.0.1:80/harvest/metadata/extract
```

### 2. Save the metadata

```bash
curl -X GET http://127.0.0.1:80/harvest/metadata/save
```

### 3. Harvest the labels

```bash
curl -X GET http://127.0.0.1:80/harvest/labels/extract
```

### 4. Save the labels

```bash
curl -X GET http://127.0.0.1:80/harvest/labels/save
```

### 5. Harvest the colors

```bash
curl -X GET http://127.0.0.1:80/harvest/colors/extract
```

### 6. Save the colors

```bash
curl -X GET http://127.0.0.1:80/harvest/colors/save
```

## Visualize service :

### TODO....

## Recommend service :

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "Make": "Canon",
      "dominant_color": "#000000",
      "tags": "['cat', 'dog']",
      "ImageWidth": "1000",
      "ImageHeight": "1000"
      }' \
      http://127.0.0.1:80/recommend/recommend
```

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "dominant_color": "#000000"
      }' \
      http://127.0.0.1:80/recommend/recommend
```

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "tags": ["cat", "dog"],
      "Orientation": 1
      }' \
      http://127.0.0.1:80/recommend/recommend
```

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "Make": "Canon",
      "ImageWidth": "1000"
      }' \
      http://127.0.0.1:80/recommend/recommend
```

```bash
curl -X POST \
    -H "Content-Type: application/json" \
  -d '{
    "tags": "['cat', 'dog']"
    }' \
    http://127.0.0.1:80/recommend/recommend
```

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
    "tags": "['cat']",
    "Make": "Canon"
    }' \
http://127.0.0.1:80/recommend/recommend

```
    
```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "tags": "['cat', 'dog']",
      "ImageWidth": "1000",
      "ImageHeight": "1000"
      }' \
      http://127.0.0.1:80/recommend/recommend
```

```bash
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
    "tags": "cat"
    }' \
http://127.0.0.1:80/recommend/recommend

```