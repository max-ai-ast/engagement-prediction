# Inference Service Deployment

This directory contains deployment and setup scripts for the engagement inference service.

## Stable Domains

The service supports stable Cloud Run domain mappings:

- stage: <https://inference-stage.greenearth.social>
- prod: <https://inference.greenearth.social>

The scripts are idempotent and safe to re-run.

## One-Time Setup Per Environment

Run GCP setup for each environment. This creates/updates:

- service account
- model storage bucket
- inference API key secret
- Cloud Run domain mapping
- Cloud DNS CNAME record (using the hardcoded zone in gcp_setup.sh)

```bash
# stage
GE_ENVIRONMENT=stage ./gcp_setup.sh

# prod
GE_ENVIRONMENT=prod ./gcp_setup.sh
```

Optional flags:

- `--inference-domain <domain>`: use a custom host
- `--disable-domain-mapping`: skip mapping/DNS setup

## Deploying Inference Service

Deploy as usual:

```bash
GE_INFERENCE_USER_TOWER_MODEL_URI=gs://your-bucket/path/to/engagement_user_tower.pt
GE_ENVIRONMENT=stage ./deploy.sh --models user-tower,post-tower --max-history-len 128
GE_ENVIRONMENT=prod ./deploy.sh --models user-tower,post-tower --max-history-len 128
```

`deploy.sh` reconciles the domain mapping after deploy (unless disabled) and prints:

- Cloud Run URL
- mapped URL readiness

## API Security

Inference endpoints are publicly routable but protected with `X-API-Key`.
The deploy script injects `GE_INFERENCE_API_KEY` from Secret Manager:

- `inference-api-key-stage`
- `inference-api-key-prod`

Keep those secrets in sync with API service secrets for cross-service calls.

## Local Development

When you want API to call a local inference instance, override API deployment with:

```bash
GE_INFERENCE_BASE_URL="http://127.0.0.1:8001" ./scripts/deploy.sh --environment stage
```

That explicit base URL override takes precedence over mapped domains.
