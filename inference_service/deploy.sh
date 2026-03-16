#!/bin/bash

# Green Earth Engagement Prediction - Inference Service Cloud Run Deployment Script
# Deploys the inference FastAPI service to Google Cloud Run using source deployment.
# The CPU Dockerfile in this directory is picked up automatically.
#
# Prerequisites: Run gcp_setup.sh first to configure the GCP environment.
# Must be run from within the inference_service/ directory.

set -e

# Configuration
GE_GCP_PROJECT_ID="${GE_GCP_PROJECT_ID:-greenearth-471522}"
GE_GCP_REGION="${GE_GCP_REGION:-us-east1}"
GE_ENVIRONMENT="${GE_ENVIRONMENT:-stage}"

# Model URI — required, no default
GE_INFERENCE_MODEL_URI="${GE_INFERENCE_MODEL_URI:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_build() {
    echo -e "${BLUE}[BUILD]${NC} $1"
}

validate_config() {
    log_info "Validating configuration..."

    if [ -z "$GE_INFERENCE_MODEL_URI" ]; then
        log_error "GE_INFERENCE_MODEL_URI is required. Set it via env var or --model-uri flag."
        log_error "Example: GE_INFERENCE_MODEL_URI=gs://my-bucket/model/ ./deploy.sh"
        exit 1
    fi

    gcloud config set project "$GE_GCP_PROJECT_ID"

    log_info "Configuration validation complete."
}

verify_vpc_connector() {
    log_info "Verifying VPC connector exists..."

    local connector_name="ingex-vpc-connector-$GE_ENVIRONMENT"

    if ! gcloud compute networks vpc-access connectors describe "$connector_name" --region="$GE_GCP_REGION" > /dev/null 2>&1; then
        log_warn "VPC connector '$connector_name' does not exist"
        log_warn "Deploying without VPC connector — service will not have internal network access"
        log_warn "Run ../ingex/ingest/scripts/gcp_setup.sh to create the VPC connector if needed"
        VPC_CONNECTOR_EXISTS=false
    else
        local connector_status
        connector_status=$(gcloud compute networks vpc-access connectors describe "$connector_name" --region="$GE_GCP_REGION" --format="value(state)" 2>/dev/null || echo "UNKNOWN")

        if [ "$connector_status" != "READY" ]; then
            log_warn "VPC connector '$connector_name' is not ready (status: $connector_status)"
            log_warn "This may cause deployment to fail. Wait a few minutes and try again."
        else
            log_info "VPC connector '$connector_name' is ready"
        fi
        VPC_CONNECTOR_EXISTS=true
    fi
}

deploy_inference_service() {
    local service_name="engagement-prediction-inference-$GE_ENVIRONMENT"
    local sa_email="engagement-prediction-sa-$GE_ENVIRONMENT@$GE_GCP_PROJECT_ID.iam.gserviceaccount.com"

    log_info "Deploying $service_name from source..."

    local deploy_cmd="gcloud run deploy $service_name"
    deploy_cmd="$deploy_cmd --source=."
    deploy_cmd="$deploy_cmd --region=$GE_GCP_REGION"
    deploy_cmd="$deploy_cmd --service-account=$sa_email"

    if [ "$VPC_CONNECTOR_EXISTS" = true ]; then
        deploy_cmd="$deploy_cmd --vpc-connector=ingex-vpc-connector-$GE_ENVIRONMENT"
        deploy_cmd="$deploy_cmd --vpc-egress=private-ranges-only"
    fi

    deploy_cmd="$deploy_cmd --ingress=internal"
    deploy_cmd="$deploy_cmd --allow-unauthenticated"

    deploy_cmd="$deploy_cmd --set-env-vars=GE_INFERENCE_MODEL_URI=$GE_INFERENCE_MODEL_URI"
    deploy_cmd="$deploy_cmd --set-env-vars=GE_INFERENCE_PREFER_CUDA=0"
    deploy_cmd="$deploy_cmd --set-env-vars=GE_INFERENCE_WARMUP=0"

    deploy_cmd="$deploy_cmd --cpu=2"
    deploy_cmd="$deploy_cmd --memory=2Gi"
    deploy_cmd="$deploy_cmd --timeout=120"
    deploy_cmd="$deploy_cmd --min-instances=0"
    deploy_cmd="$deploy_cmd --max-instances=1"

    log_build "Executing: $deploy_cmd"
    eval "$deploy_cmd"

    log_info "✓ $service_name deployed successfully"

    local service_url
    service_url=$(gcloud run services describe "$service_name" --region="$GE_GCP_REGION" --format="value(status.url)")
    log_info "Service URL: $service_url"
    log_info "(Note: --ingress=internal — only reachable from within the VPC)"
}

main() {
    log_info "Starting engagement prediction inference service deployment..."
    log_info "Project:     $GE_GCP_PROJECT_ID"
    log_info "Region:      $GE_GCP_REGION"
    log_info "Environment: $GE_ENVIRONMENT"
    log_info "Model URI:   $GE_INFERENCE_MODEL_URI"

    validate_config
    verify_vpc_connector
    deploy_inference_service

    log_info "Deployment complete!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            GE_GCP_PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            GE_GCP_REGION="$2"
            shift 2
            ;;
        --environment)
            GE_ENVIRONMENT="$2"
            shift 2
            ;;
        --model-uri)
            GE_INFERENCE_MODEL_URI="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --project-id ID          GCP project ID (default: greenearth-471522)"
            echo "  --region REGION          GCP region (default: us-east1)"
            echo "  --environment ENV        Environment name (default: stage)"
            echo "  --model-uri URI          GCS URI for the model (required)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  GE_GCP_PROJECT_ID        Same as --project-id"
            echo "  GE_GCP_REGION            Same as --region"
            echo "  GE_ENVIRONMENT           Same as --environment"
            echo "  GE_INFERENCE_MODEL_URI   Same as --model-uri (required)"
            echo ""
            echo "Examples:"
            echo "  GE_INFERENCE_MODEL_URI=gs://my-bucket/model/ $0 --environment stage"
            echo "  $0 --environment stage --model-uri gs://my-bucket/model/"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

main
