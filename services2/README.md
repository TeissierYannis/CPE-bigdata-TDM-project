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