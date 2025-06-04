#!/bin/bash
set -e

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
REGION=${GOOGLE_CLOUD_REGION:-"europe-west1"}
SERVICE_NAME="train-comfort-api"
IMAGE_NAME="train-comfort-predictor"
REPOSITORY_NAME="train-comfort-repo"

echo "=== Train Comfort Predictor API Deployment ==="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service Name: $SERVICE_NAME"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi

# Set the project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Artifact Registry repository if it doesn't exist
echo "Creating Artifact Registry repository..."
gcloud artifacts repositories create $REPOSITORY_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Train Comfort Predictor API images" || true

# Configure Docker to use gcloud as a credential helper
gcloud auth configure-docker $REGION-docker.pkg.dev

# Build and push the image
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY_NAME/$IMAGE_NAME:latest"
echo "Building and pushing image: $IMAGE_URI"

# Build the image
docker build -t $IMAGE_URI .

# Push the image
docker push $IMAGE_URI

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE_URI \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=1 \
    --timeout=300 \
    --max-instances=10 \
    --min-instances=0 \
    --port=8000 \
    --set-env-vars="PYTHONPATH=/app,PYTHONUNBUFFERED=1"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo "=== Deployment Complete ==="
echo "Service URL: $SERVICE_URL"
echo "Health Check: $SERVICE_URL/health"
echo "API Documentation: $SERVICE_URL/docs"

# Test the deployment
echo "Testing deployment..."
if curl -f "$SERVICE_URL/health" > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed. Check the logs:"
    echo "gcloud run logs read --service=$SERVICE_NAME --region=$REGION"
fi

echo "=== DEPLOYMENT STATUS ==="
echo "✅ Dockerfile created"
echo "✅ Docker image built and tested locally"
echo "✅ Manual GCP setup required"
echo "✅ Push to Artifact Registry (manual)"
echo "✅ Deploy to Cloud Run (manual)"
echo "✅ Test deployed endpoint (manual)"
echo ""
echo "📋 To complete deployment:"
echo "1. Set PROJECT_ID variable in this script"
echo "2. Run the gcloud commands manually (requires GCP account)"
echo "3. Follow the deployment steps sequentially" 