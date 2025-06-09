# Deployment Guide for Train Comfort Predictor API

This guide covers deploying the Train Comfort Predictor API to Google Cloud Run using Artifact Registry.

## Prerequisites

1. **Google Cloud Platform Account**
   - Active GCP project with billing enabled
   - Project Owner or Editor permissions

2. **Local Tools**
   - Docker installed and running
   - gcloud CLI installed and authenticated
   - Python 3.9+ with virtual environment

3. **Application Requirements**
   - Model trained: `python src/xgboost_model_training.py`
   - API database created: `api/train_comfort_api_lookups.sqlite`
   - All tests passing: `python test_api.py`

## Environment Setup

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file with your values:**
   ```bash
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_REGION=europe-west1
   ```

3. **Source environment variables:**
   ```bash
   source .env
   ```

## Deployment Steps

### 1. Authenticate with Google Cloud

```bash
# Login to Google Cloud
gcloud auth login

# Set default project
gcloud config set project $GOOGLE_CLOUD_PROJECT
```

### 2. Run the Deployment Script

```bash
# Make script executable (if needed)
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

The script will:
- Enable required Google Cloud APIs
- Create Artifact Registry repository
- Build and push Docker image
- Deploy to Cloud Run
- Test the deployment

### 3. Manual Deployment (Alternative)

If you prefer manual deployment:

```bash
# Enable APIs
gcloud services enable cloudbuild.googleapis.com run.googleapis.com artifactregistry.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create train-comfort-repo \
    --repository-format=docker \
    --location=$GOOGLE_CLOUD_REGION

# Configure Docker
gcloud auth configure-docker $GOOGLE_CLOUD_REGION-docker.pkg.dev

# Build and push image
docker build -t $GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/train-comfort-repo/train-comfort-predictor:latest .
docker push $GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/train-comfort-repo/train-comfort-predictor:latest

# Deploy to Cloud Run
gcloud run deploy train-comfort-api \
    --image=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/train-comfort-repo/train-comfort-predictor:latest \
    --region=$GOOGLE_CLOUD_REGION \
    --platform=managed \
    --allow-unauthenticated \
    --memory=2Gi \
    --cpu=1 \
    --timeout=300 \
    --max-instances=10 \
    --port=8000
```

## Post-Deployment

### Test the Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe train-comfort-api --region=$GOOGLE_CLOUD_REGION --format="value(status.url)")

# Test health endpoint
curl $SERVICE_URL/health

# Test with the test script
python test_api.py $SERVICE_URL
```

### Monitor the Service

```bash
# View logs
gcloud run logs read --service=train-comfort-api --region=$GOOGLE_CLOUD_REGION

# View metrics in Cloud Console
echo "https://console.cloud.google.com/run/detail/$GOOGLE_CLOUD_REGION/train-comfort-api/metrics?project=$GOOGLE_CLOUD_PROJECT"
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   gcloud auth application-default login
   ```

2. **Docker Permission Error**
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. **Memory Issues**
   - Increase memory allocation in deploy script: `--memory=4Gi`

4. **Timeout Issues**
   - Increase timeout in deploy script: `--timeout=600`

### Rollback

To rollback to a previous version:
```bash
gcloud run services update-traffic train-comfort-api \
    --region=$GOOGLE_CLOUD_REGION \
    --to-revisions=PREV=100
```

## Security Considerations

- The API is deployed with `--allow-unauthenticated` for public access
- To add authentication, remove this flag and configure IAM
- Consider using Cloud Armor for DDoS protection
- Enable Cloud Audit Logs for monitoring

## Cost Management

- Set up budget alerts in GCP Console
- Configure auto-scaling limits appropriately
- Consider using Cloud Scheduler to scale down during off-hours

## Next Steps

1. Set up CI/CD with Cloud Build
2. Configure custom domain
3. Add monitoring and alerting
4. Implement caching with Cloud CDN