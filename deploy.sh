#!/bin/bash
set -e

# Load environment variables if .env exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"your-project-id"}
REGION=${GOOGLE_CLOUD_REGION:-"europe-west1"}
SERVICE_NAME=${SERVICE_NAME:-"train-comfort-api"}
IMAGE_NAME=${IMAGE_NAME:-"train-comfort-predictor"}
REPOSITORY_NAME=${REPOSITORY_NAME:-"train-comfort-repo"}

# Check if PROJECT_ID is still the default
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Error: GOOGLE_CLOUD_PROJECT not set"
    echo "Please either:"
    echo "  1. Create a .env file with GOOGLE_CLOUD_PROJECT=your-actual-project-id"
    echo "  2. Export the variable: export GOOGLE_CLOUD_PROJECT=your-actual-project-id"
    exit 1
fi

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

# Detect if we're on an ARM Mac
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected ARM64 architecture (Apple Silicon). Using buildx for AMD64 build..."
    
    # Ensure buildx is set up
    docker buildx create --name gcr-builder --use 2>/dev/null || docker buildx use gcr-builder
    
    # Build and push in one step for AMD64
    docker buildx build --platform linux/amd64 -t $IMAGE_URI . --push
else
    # Build the image normally
    docker build -t $IMAGE_URI .
    
    # Push the image
    docker push $IMAGE_URI
fi

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
    --port=8080 \
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

echo "=== DEPLOYMENT COMPLETE ==="
echo ""
echo "📋 Next steps:"
echo "1. Monitor service: gcloud run logs read --service=$SERVICE_NAME --region=$REGION"
echo "2. View metrics in Cloud Console"
echo "3. Set up custom domain (optional)"
echo "4. Configure monitoring alerts (recommended)" 