To make IAM Service Account key (for example):  
`gcloud config set project greenearth-471522`  
`gcloud iam service-accounts keys create /path/to/sa-key.json --iam-account=engagement-prediction-sa-test@greenearth-471522.iam.gserviceaccount.com`  
`chmod 600 /path/to/sa-key.json`  
If you want to use the key in the current environment:  
`export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json`  

To run the docker image once built (for example):  
`sudo docker run --rm -p 8080:8080 --gpus all --env-file $PWD/.env -v "/path/to/sa-key.json:/var/secrets/google/sa.json:ro" inference-service:git-<GIT_SHA>`  
(This will mount the google application credentials on the container. You can set the path to it in your `.env` file).

For local development, if you don't want to re-build the image every time you update `app.py`, add these to the above `docker run` command:  
`UVICORN_RELOAD=1 -e UVICORN_WORKERS=1 -v "$(pwd):/app:ro`