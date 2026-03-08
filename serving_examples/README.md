# ClearML Serving + Triton (GPU) examples

This directory contains a helper script and Docker Compose files for spinning up a local ClearML Serving inference endpoint backed by Triton, then registering a ClearML model onto an endpoint.

## Prerequisites

- Docker + Docker Compose v2 (`docker compose`)
- `clearml-serving` CLI available in your `PATH` (used by the script to create the service + register the model)
- ClearML credentials (for both the local `clearml-serving` CLI and the containers)

### GPU host requirements (Linux)

If you’re running the GPU compose file, you may need NVIDIA’s container runtime tooling:

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Monitor GPU usage:

```bash
watch -n 0.5 nvidia-smi
```

## Configure environment

The compose files read ClearML settings from `serving_examples/docker.env`.

1. Create `serving_examples/docker.env` from the template:

```bash
cp serving_examples/docker.env.template serving_examples/docker.env
```

2. Edit `serving_examples/docker.env` and set:
   - `CLEARML_API_ACCESS_KEY`
   - `CLEARML_API_SECRET_KEY`
   - (optional) `CLEARML_*_HOST` if you’re not using ClearML Hosted

Note: the script will overwrite `CLEARML_SERVING_TASK_ID` in `serving_examples/docker.env` after it creates a new serving service.

## Create a service and add a model endpoint

The script is `serving_examples/create_clearml_triton_and_up.sh`.

From the repo root:

```bash
./serving_examples/create_clearml_triton_and_up.sh --model-type mlp --model-id <CLEARML_MODEL_ID>
```

What it does:

- `clearml-serving create --name ...` (creates a new ClearML Serving service)
- Writes the resulting service id to `serving_examples/docker.env` as `CLEARML_SERVING_TASK_ID=...`
- Runs `sudo docker compose ... up -d` using `serving_examples/docker-compose-triton-gpu.yml`
- Registers the model to an endpoint via `clearml-serving --id ... model add --engine triton ...`
- Streams docker logs (Ctrl+C stops log streaming; containers keep running)

### Model types and endpoints

- `--model-type` must be one of: `mlp`, `post`, `user`
- `--endpoint` defaults to the model type (e.g. `mlp`), so the URL will typically be:
  - `http://127.0.0.1:8080/serve/<endpoint>`
- `--preprocess` can override the default preprocess script (defaults are under `serving_examples/`)

## Hitting the endpoint (examples)

- `python serving_examples/hit_inference_endpoint_example_mlp.py`
- `python serving_examples/hit_inference_endpoint_example_post.py`
- `python serving_examples/hit_inference_endpoint_example_user.py`

## Re-adding a model without recreating the service

If you already have a service id (stored in `serving_examples/docker.env`), you can skip service creation + docker compose and only run the model registration step:

```bash
./serving_examples/create_clearml_triton_and_up.sh --only-model-add --model-type mlp --model-id <CLEARML_MODEL_ID>
```

To explicitly pass a service id:

```bash
./serving_examples/create_clearml_triton_and_up.sh --only-model-add --serving-id <SERVING_TASK_ID> --model-type mlp --model-id <CLEARML_MODEL_ID>
```

## Stopping the stack

```bash
sudo docker compose --env-file serving_examples/docker.env -f serving_examples/docker-compose-triton-gpu.yml down
```
