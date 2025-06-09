# Deployment Troubleshooting Guide

## Common Cloud Run Deployment Issues

### 1. Port Configuration
**Issue**: Container fails to start with "failed to listen on PORT" error

**Solution**: Cloud Run requires apps to listen on port 8080 by default
- The app now reads the PORT environment variable (defaults to 8080)
- All configurations have been updated to use port 8080
- The Dockerfile exposes port 8080 and sets it in the CMD

### 2. CPU Architecture Mismatch
**Issue**: Container crashes or behaves unexpectedly on Cloud Run

**Cause**: Building Docker images on Apple Silicon (M1/M2) creates ARM64 images, but Cloud Run needs AMD64

**Solution**: The Dockerfile now specifies the platform explicitly:
```dockerfile
FROM --platform=linux/amd64 python:3.11-slim
```

**Build command for M1/M2 Macs**:
```bash
# Force AMD64 build on ARM Macs
docker buildx build --platform linux/amd64 -t train-comfort-api .

# Or set Docker to use buildx by default
docker buildx create --use
docker buildx build --platform linux/amd64 -t $IMAGE_URI . --push
```

### 3. Path Resolution Issues
**Issue**: Model files or database not found in container

**Solution**: Use absolute paths based on the script location:
- Models are loaded from `/app/models/`
- Database is loaded from `/app/api/`
- All paths are resolved using `os.path.join()` and `os.path.abspath()`

## Pre-deployment Checklist

1. **Verify model files exist**:
   ```bash
   ls -la models/*.joblib
   ```

2. **Verify database exists**:
   ```bash
   ls -la api/train_comfort_api_lookups.sqlite
   ```

3. **Test locally with correct architecture**:
   ```bash
   # Build for AMD64
   docker build --platform linux/amd64 -t train-comfort-api-test .
   
   # Run with Cloud Run port
   docker run -p 8080:8080 -e PORT=8080 train-comfort-api-test
   
   # Test health endpoint
   curl http://localhost:8080/health
   ```

4. **Check Docker daemon architecture**:
   ```bash
   docker version
   # Look for "OS/Arch" in both Client and Server sections
   ```

## Deployment Commands for M1/M2 Macs

```bash
# Ensure buildx is available
docker buildx create --name mybuilder --use
docker buildx inspect --bootstrap

# Build and push in one command
docker buildx build \
  --platform linux/amd64 \
  -t $REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/train-comfort-repo/train-comfort-predictor:latest \
  --push .

# Then deploy to Cloud Run
gcloud run deploy train-comfort-api \
  --image=$REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/train-comfort-repo/train-comfort-predictor:latest \
  --region=$REGION \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --port=8080
```

## Debugging Failed Deployments

1. **Check Cloud Run logs**:
   ```bash
   gcloud run logs read --service=train-comfort-api --region=$REGION --limit=50
   ```

2. **Check revision status**:
   ```bash
   gcloud run revisions list --service=train-comfort-api --region=$REGION
   ```

3. **Describe the service**:
   ```bash
   gcloud run services describe train-comfort-api --region=$REGION
   ```

4. **Local debugging with exact Cloud Run environment**:
   ```bash
   # Run with Cloud Run's environment variables
   docker run --rm \
     -e PORT=8080 \
     -e K_SERVICE=train-comfort-api \
     -e K_REVISION=train-comfort-api-00001 \
     -e K_CONFIGURATION=train-comfort-api \
     -p 8080:8080 \
     train-comfort-api-test
   ```

## Quick Fix Summary

The main fixes applied to resolve deployment issues:

1. **Port**: Changed from 8000/8001 to 8080 everywhere
2. **Architecture**: Added `--platform=linux/amd64` to Dockerfile
3. **Paths**: Made all file paths absolute using `os.path.join()`
4. **Environment**: App reads PORT from environment variable

With these fixes, the deployment should work correctly on Cloud Run.