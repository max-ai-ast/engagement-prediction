#!/bin/bash

# Green Earth Engagement Prediction - GCP Environment Setup Script
# This script sets up the GCP environment for the first time
# Run this once per environment (test, stage, prod])

set -e

# Configuration
GE_GCP_PROJECT_ID="${GE_GCP_PROJECT_ID:-greenearth-471522}"
GE_GCP_REGION="${GE_GCP_REGION:-us-east1}"
GE_ENVIRONMENT="${GE_ENVIRONMENT:-stage}"  # TODO: change default when we have more environments
GE_ENABLE_INFERENCE_DOMAIN_MAPPING="${GE_ENABLE_INFERENCE_DOMAIN_MAPPING:-true}"
GE_INFERENCE_DOMAIN="${GE_INFERENCE_DOMAIN:-}"
# TODO: Replace with your actual managed zone name if different.
GE_CLOUD_DNS_ZONE="greenearth-social"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

default_inference_domain() {
    if [ "$GE_ENVIRONMENT" = "prod" ]; then
        echo "inference.greenearth.social"
    else
        echo "inference-stage.greenearth.social"
    fi
}

resolve_inference_domain() {
    if [ -n "$GE_INFERENCE_DOMAIN" ]; then
        echo "$GE_INFERENCE_DOMAIN"
        return
    fi
    echo "$(default_inference_domain)"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi

    if ! command -v python3 &> /dev/null; then
        log_error "python3 is required for domain mapping status checks. Please install it first."
        exit 1
    fi

    # Check if user is logged in
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_error "Please log in to gcloud first: gcloud auth login"
        exit 1
    fi

    log_info "Prerequisites check complete."
}

validate_config() {
    log_info "Validating configuration..."

    if [ "$GE_GCP_PROJECT_ID" = "your-project-id" ]; then
        log_error "Please set GE_GCP_PROJECT_ID environment variable or update the script"
        exit 1
    fi

    log_info "Configuration validation complete."
}

setup_gcp_project() {
    log_info "Setting up GCP project: $GE_GCP_PROJECT_ID"

    # Set the project
    gcloud config set project "$GE_GCP_PROJECT_ID"

    # Enable required APIs
    log_info "Enabling required GCP APIs..."
    gcloud services enable \
        storage.googleapis.com \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        artifactregistry.googleapis.com \
        vpcaccess.googleapis.com \
        compute.googleapis.com \
        secretmanager.googleapis.com \
        dns.googleapis.com

    log_info "GCP project setup complete."
}

cloud_run_service_exists() {
    local service_name="engagement-prediction-inference-$GE_ENVIRONMENT"

    gcloud run services describe "$service_name" \
        --region="$GE_GCP_REGION" \
        --project="$GE_GCP_PROJECT_ID" \
        > /dev/null 2>&1
}

get_domain_mapping_condition_status() {
    local domain="$1"
    local condition_type="$2"

    gcloud beta run domain-mappings describe --domain="$domain" \
        --region="$GE_GCP_REGION" \
        --project="$GE_GCP_PROJECT_ID" \
        --format=json 2>/dev/null | python3 -c '
import json
import sys

condition_type = sys.argv[1]

try:
    payload = json.load(sys.stdin)
except json.JSONDecodeError:
    print("")
    raise SystemExit(0)

for condition in payload.get("status", {}).get("conditions", []):
    if condition.get("type") == condition_type:
        print(condition.get("status", ""))
        break
else:
    print("")
' "$condition_type"
}

ensure_domain_mapping() {
    if [ "$GE_ENABLE_INFERENCE_DOMAIN_MAPPING" != "true" ]; then
        log_info "Skipping domain mapping setup (GE_ENABLE_INFERENCE_DOMAIN_MAPPING=$GE_ENABLE_INFERENCE_DOMAIN_MAPPING)"
        return
    fi

    local domain
    domain="$(resolve_inference_domain)"
    local service_name="engagement-prediction-inference-$GE_ENVIRONMENT"

    if ! cloud_run_service_exists; then
        log_warn "Cloud Run service '$service_name' does not exist yet"
        log_warn "Skipping domain mapping creation for now; deploy.sh will reconcile it after first deploy"
        return
    fi

    log_info "Ensuring Cloud Run domain mapping exists for $domain -> $service_name"

    if gcloud beta run domain-mappings describe --domain="$domain" --region="$GE_GCP_REGION" --project="$GE_GCP_PROJECT_ID" > /dev/null 2>&1; then
        log_info "Domain mapping already exists: $domain"
    else
        gcloud beta run domain-mappings create \
            --service="$service_name" \
            --domain="$domain" \
            --region="$GE_GCP_REGION" \
            --project="$GE_GCP_PROJECT_ID"
        log_info "Created domain mapping: $domain"
    fi
}

ensure_domain_dns_record() {
    if [ "$GE_ENABLE_INFERENCE_DOMAIN_MAPPING" != "true" ]; then
        return
    fi

    local domain
    domain="$(resolve_inference_domain)"
    local dns_name="${domain}."
    local cname_target="ghs.googlehosted.com."

    log_info "Ensuring Cloud DNS CNAME record exists for $dns_name in zone $GE_CLOUD_DNS_ZONE"

    if gcloud dns record-sets describe "$dns_name" \
        --type="CNAME" \
        --zone="$GE_CLOUD_DNS_ZONE" \
        --project="$GE_GCP_PROJECT_ID" > /dev/null 2>&1; then
        log_info "DNS CNAME already exists: $dns_name"
    else
        gcloud dns record-sets create "$dns_name" \
            --type="CNAME" \
            --ttl="300" \
            --rrdatas="$cname_target" \
            --zone="$GE_CLOUD_DNS_ZONE" \
            --project="$GE_GCP_PROJECT_ID"
        log_info "Created DNS CNAME: $dns_name -> $cname_target"
    fi
}

wait_for_domain_mapping_ready() {
    if [ "$GE_ENABLE_INFERENCE_DOMAIN_MAPPING" != "true" ]; then
        return
    fi

    local domain
    domain="$(resolve_inference_domain)"

    if ! gcloud beta run domain-mappings describe --domain="$domain" \
        --region="$GE_GCP_REGION" \
        --project="$GE_GCP_PROJECT_ID" > /dev/null 2>&1; then
        log_info "Domain mapping does not exist yet for $domain; skipping readiness check"
        return
    fi

    log_info "Checking domain mapping readiness for $domain"
    log_info "Certificate provisioning can take several minutes after DNS is in place"

    local max_attempts=20
    local attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
        local ready
        ready=$(get_domain_mapping_condition_status "$domain" "Ready")

        if [ "$ready" = "True" ]; then
            log_info "Domain mapping is Ready: https://$domain"
            return
        fi

        log_info "Domain mapping not ready yet (attempt $attempt/$max_attempts)"
        sleep 15
        attempt=$((attempt + 1))
    done

    log_warn "Domain mapping is still not Ready: $domain"
    log_warn "Run: gcloud beta run domain-mappings describe --domain=$domain --region=$GE_GCP_REGION --project=$GE_GCP_PROJECT_ID"
}

create_service_account() {
    log_info "Creating service account for engagement prediction model management..."

    SA_NAME="engagement-prediction-sa-$GE_ENVIRONMENT"
    SA_EMAIL="$SA_NAME@$GE_GCP_PROJECT_ID.iam.gserviceaccount.com"

    # Create service account
    if ! gcloud iam service-accounts describe "$SA_EMAIL" > /dev/null 2>&1; then
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="Engagement Prediction Service Account ($GE_ENVIRONMENT)" \
            --description="Service account for engagement prediction model management in $GE_ENVIRONMENT"
        log_info "Service account created: $SA_EMAIL"
    else
        log_info "Service account already exists: $SA_EMAIL"
    fi
}

create_engagement_prediction_model_storage() {
    log_info "Setting up storage bucket for engagement prediction models..."

    BUCKET_NAME="$GE_GCP_PROJECT_ID-engagement-prediction-model-$GE_ENVIRONMENT"

    if ! gsutil ls -b gs://"$BUCKET_NAME" > /dev/null 2>&1; then
        gsutil mb -l "$GE_GCP_REGION" gs://"$BUCKET_NAME"
        log_info "Engagement prediction model storage bucket created: $BUCKET_NAME"
    else
        log_info "Engagement prediction model storage bucket already exists: $BUCKET_NAME"
    fi

    # Grant engagement prediction service account objectAdmin permission on this bucket
    ENG_PRED_SA_EMAIL="engagement-prediction-sa-$GE_ENVIRONMENT@$GE_GCP_PROJECT_ID.iam.gserviceaccount.com"
    gsutil iam ch serviceAccount:"$ENG_PRED_SA_EMAIL":objectAdmin gs://"$BUCKET_NAME"
    log_info "Granted objectAdmin to engagement prediction service account for bucket: $BUCKET_NAME"
}

create_api_key_secret() {
    log_info "Setting up API key secret in Secret Manager..."

    local secret_name="inference-api-key-$GE_ENVIRONMENT"
    local sa_email="engagement-prediction-sa-$GE_ENVIRONMENT@$GE_GCP_PROJECT_ID.iam.gserviceaccount.com"

    if gcloud secrets describe "$secret_name" > /dev/null 2>&1; then
        log_info "Secret '$secret_name' already exists — skipping creation"
    else
        gcloud secrets create "$secret_name" --replication-policy=automatic
        local api_key
        api_key=$(openssl rand -hex 32)
        echo -n "$api_key" | gcloud secrets versions add "$secret_name" --data-file=-
        log_info "Secret '$secret_name' created"
        log_info "API key (save this now — it will not be shown again):"
        echo ""
        echo "  $api_key"
        echo ""
    fi

    gcloud secrets add-iam-policy-binding "$secret_name" \
        --member="serviceAccount:$sa_email" \
        --role="roles/secretmanager.secretAccessor" \
        > /dev/null
    log_info "Granted secretAccessor to $sa_email on '$secret_name'"
}

check_vpc_connector() {
    log_info "Checking for VPC connector..."

    local connector_name="ingex-vpc-connector-$GE_ENVIRONMENT"

    if gcloud compute networks vpc-access connectors describe "$connector_name" --region="$GE_GCP_REGION" > /dev/null 2>&1; then
        log_info "VPC connector '$connector_name' already exists"
        log_info "Inference service will be able to use this for internal network access"
    else
        log_warn "VPC connector '$connector_name' does not exist"
        log_warn "If you need internal network access, run:"
        log_warn "  cd ../../ingex/ingest && ./scripts/gcp_setup.sh"
        log_warn ""
        log_warn "The inference service can still be deployed without VPC connector"
    fi
}

main() {
    echo "=========================================================="
    echo "Green Earth Engagement Prediction - GCP Environment Setup"
    echo "Environment: $GE_ENVIRONMENT"
    echo "Project: $GE_GCP_PROJECT_ID"
    echo "Region: $GE_GCP_REGION"
    echo "=========================================================="
    echo

    check_prerequisites
    validate_config
    setup_gcp_project
    create_service_account
    create_engagement_prediction_model_storage
    create_api_key_secret
    check_vpc_connector
    ensure_domain_mapping
    ensure_domain_dns_record
    wait_for_domain_mapping_ready

    log_info "Environment setup complete!"
    echo
    echo "Next steps:"
    echo "1. Run 'inference_service/deploy.sh' to deploy the inference service to Cloud Run"
    echo "2. Re-run this script or let deploy.sh reconcile domain mapping after the first deploy"
    echo "3. Check Cloud Run console to verify the service is running"
    echo "4. Verify domain mapping status and certificate provisioning"
    echo
    echo "Important notes:"
    echo "- Model files are stored in: gs://$GE_GCP_PROJECT_ID-engagement-prediction-model-$GE_ENVIRONMENT"
    echo "- Service account: engagement-prediction-sa-$GE_ENVIRONMENT@$GE_GCP_PROJECT_ID.iam.gserviceaccount.com"
    echo "- Service name: engagement-prediction-inference-$GE_ENVIRONMENT"
    echo "- API key secret: inference-api-key-$GE_ENVIRONMENT (in Secret Manager)"
    if [ "$GE_ENABLE_INFERENCE_DOMAIN_MAPPING" = "true" ]; then
        echo "- Inference domain: $(resolve_inference_domain)"
        echo "- Cloud DNS zone: $GE_CLOUD_DNS_ZONE"
    fi
    echo
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
        --inference-domain)
            GE_INFERENCE_DOMAIN="$2"
            shift 2
            ;;
        --disable-domain-mapping)
            GE_ENABLE_INFERENCE_DOMAIN_MAPPING="false"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --project-id ID              GCP project ID"
            echo "  --region REGION              GCP region (default: us-east1)"
            echo "  --environment ENV            Environment name (default: stage)"
            echo "  --inference-domain DOMAIN    Custom domain for inference service"
            echo "  --disable-domain-mapping     Skip Cloud Run domain mapping + DNS setup"
            echo "  --help                       Show this help message"
            echo
            echo "The script is idempotent and safe to re-run to ensure correct configuration."
            echo
            echo "Environment variables:"
            echo "  GE_GCP_PROJECT_ID, GE_GCP_REGION, GE_ENVIRONMENT"
            echo "  GE_ENABLE_INFERENCE_DOMAIN_MAPPING, GE_INFERENCE_DOMAIN"
            echo
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
