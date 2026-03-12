#!/bin/bash

# Green Earth Engagement Prediction - GCP Environment Setup Script
# This script sets up the GCP environment for the first time
# Run this once per environment (test, stage, prod])

set -e

# Configuration
GE_GCP_PROJECT_ID="${GE_GCP_PROJECT_ID:-greenearth-471522}"
GE_GCP_REGION="${GE_GCP_REGION:-us-east1}"
GE_ENVIRONMENT="${GE_ENVIRONMENT:-stage}"  # TODO: change default when we have more environments

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

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
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
        storage.googleapis.com

    log_info "GCP project setup complete."
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

    log_info "Environment setup complete!"
    echo
    echo "Next steps:"
    echo "1. Run './scripts/deploy.sh' to build and deploy your services"
    echo "2. Check Cloud Run console to verify services are running"
    echo "3. Monitor logs for any issues"
    echo
    echo "Important notes:"
    echo "- Model files are stored in: gs://$GE_GCP_PROJECT_ID-engagement-prediction-model-$GE_ENVIRONMENT"
    echo "- Service account: engagement-prediction-sa-$GE_ENVIRONMENT@$GE_GCP_PROJECT_ID.iam.gserviceaccount.com"
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --project-id ID              GCP project ID"
            echo "  --region REGION              GCP region (default: us-east1)"
            echo "  --environment ENV            Environment name (default: stage)"
            echo "  --help                       Show this help message"
            echo
            echo "The script is idempotent and safe to re-run to ensure correct configuration."
            echo
            echo "Environment variables:"
            echo "  GE_GCP_PROJECT_ID, GE_GCP_REGION, GE_ENVIRONMENT"
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
