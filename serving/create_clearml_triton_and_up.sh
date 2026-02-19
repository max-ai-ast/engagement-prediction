#!/usr/bin/env bash
set -euo pipefail

service_name="clearml triton server"
model_id=""
endpoint="mlp"
preprocess_path="serving/preprocess_mlp.py"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
env_file="${script_dir}/docker.env"
compose_file="${script_dir}/docker-compose-triton.yml"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") --model-id <model_id> [options]

Options:
  --service-name <name>   ClearML Serving service name (default: "$service_name")
  --endpoint <endpoint>   Endpoint name (default: "$endpoint")
  --preprocess <path>     Preprocess script path (default: "$preprocess_path")
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id)
      model_id="${2:-}"
      shift 2
      ;;
    --service-name)
      service_name="${2:-}"
      shift 2
      ;;
    --endpoint)
      endpoint="${2:-}"
      shift 2
      ;;
    --preprocess)
      preprocess_path="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$model_id" ]]; then
  echo "error: --model-id is required" >&2
  usage >&2
  exit 2
fi

if ! command -v clearml-serving >/dev/null 2>&1; then
  echo "error: clearml-serving not found in PATH" >&2
  exit 1
fi

if [[ ! -f "$env_file" ]]; then
  echo "error: env file not found: $env_file" >&2
  exit 1
fi

if [[ ! -f "$compose_file" ]]; then
  echo "error: docker compose file not found: $compose_file" >&2
  exit 1
fi

if [[ "$preprocess_path" != /* ]]; then
  preprocess_path="${repo_root}/${preprocess_path}"
fi
if [[ ! -f "$preprocess_path" ]]; then
  echo "error: preprocess script not found: $preprocess_path" >&2
  exit 1
fi

log "Creating ClearML Serving service: name=\"$service_name\""
log "+ clearml-serving create --name \"$service_name\""
create_output="$(clearml-serving create --name "$service_name" 2>&1)"
echo "$create_output"

serving_id=""
if [[ "$create_output" =~ id=([^[:space:]]+) ]]; then
  serving_id="${BASH_REMATCH[1]}"
else
  echo "$create_output" >&2
  echo "error: could not parse serving id from clearml-serving output (expected 'id=...')" >&2
  exit 1
fi

log "Parsed service id: $serving_id"
log "Updating env file: $env_file"

tmp_file="$(mktemp)"
awk -v id="$serving_id" '
  /^CLEARML_SERVING_TASK_ID=/ { next }
  { print }
  END { print "CLEARML_SERVING_TASK_ID=\"" id "\"" }
' "$env_file" > "$tmp_file"
mv "$tmp_file" "$env_file"

log "Wrote: CLEARML_SERVING_TASK_ID=\"$serving_id\""

log "Starting docker compose (detached)"
log "+ sudo docker compose --env-file \"$env_file\" -f \"$compose_file\" up -d"
sudo docker compose --env-file "$env_file" -f "$compose_file" up -d

wait_for_container_running() {
  local container_name="$1"
  local timeout_secs="${2:-120}"
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if sudo docker inspect -f '{{.State.Running}}' "$container_name" >/dev/null 2>&1; then
      if [[ "$(sudo docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null || true)" == "true" ]]; then
        return 0
      fi
    fi
    if (( "$(date +%s)" - start_ts >= timeout_secs )); then
      return 1
    fi
    sleep 1
  done
}

log "Waiting for containers to be running"
if ! wait_for_container_running "clearml-serving-inference" 180; then
  log "warning: clearml-serving-inference not running after 180s; continuing anyway"
fi
if ! wait_for_container_running "clearml-serving-triton" 180; then
  log "warning: clearml-serving-triton not running after 180s; continuing anyway"
fi

log "Registering model on service"
log "+ clearml-serving --id \"$serving_id\" model add --engine triton --endpoint \"$endpoint\" --model-id \"$model_id\" ..."
clearml-serving --id "$serving_id" model add \
  --engine triton \
  --endpoint "$endpoint" \
  --model-id "$model_id" \
  --input-size "[-1,768]" \
  --input-name features \
  --input-type float32 \
  --output-size "[-1]" \
  --output-type float32 \
  --output-name probs \
  --preprocess "$preprocess_path"

log "Streaming docker logs (Ctrl+C to stop logs; containers keep running)"
log "+ sudo docker compose --env-file \"$env_file\" -f \"$compose_file\" logs -f"
sudo docker compose --env-file "$env_file" -f "$compose_file" logs -f
