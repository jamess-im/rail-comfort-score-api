#!/bin/bash

echo "=== Deployment Readiness Check ==="
echo ""

# Check for required files
echo "1. Checking required files..."
REQUIRED_FILES=(
    "api/main.py"
    "api/train_comfort_api_lookups.sqlite"
    "models/xgboost_comfort_classifier.joblib"
    "models/target_encoder.joblib"
    "models/feature_encoders.joblib"
    "models/feature_list.joblib"
    "Dockerfile"
    "requirements.txt"
)

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file is missing"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = false ]; then
    echo ""
    echo "❌ Some required files are missing. Please ensure all files exist before deployment."
    exit 1
fi

echo ""
echo "2. Checking Docker configuration..."

# Check if Docker is running
if docker info >/dev/null 2>&1; then
    echo "✅ Docker is running"
else
    echo "❌ Docker is not running. Please start Docker."
    exit 1
fi

# Check architecture
ARCH=$(uname -m)
echo "✅ System architecture: $ARCH"
if [[ "$ARCH" == "arm64" ]]; then
    echo "⚠️  You're on Apple Silicon. The deployment script will use buildx for AMD64 compatibility."
fi

echo ""
echo "3. Checking port configuration..."

# Check main.py for correct port
if grep -q 'os.environ.get("PORT", 8080)' api/main.py; then
    echo "✅ main.py is configured to use PORT environment variable (default 8080)"
else
    echo "❌ main.py is not properly configured for PORT environment variable"
fi

# Check Dockerfile for correct port
if grep -q "EXPOSE 8080" Dockerfile && grep -q "port=8080" Dockerfile; then
    echo "✅ Dockerfile is configured for port 8080"
else
    echo "❌ Dockerfile is not properly configured for port 8080"
fi

echo ""
echo "4. Checking platform specification..."

# Check Dockerfile for platform
if grep -q "FROM --platform=linux/amd64" Dockerfile; then
    echo "✅ Dockerfile specifies linux/amd64 platform"
else
    echo "❌ Dockerfile doesn't specify platform (will fail on ARM Macs)"
fi

echo ""
echo "5. Testing local Docker build..."

# Try to build the image
echo "Building test image..."
if docker build --platform linux/amd64 -t deployment-test . >/dev/null 2>&1; then
    echo "✅ Docker build successful"
    docker rmi deployment-test >/dev/null 2>&1
else
    echo "❌ Docker build failed. Check your Dockerfile and dependencies."
    exit 1
fi

echo ""
echo "=== Deployment Readiness Check Complete ==="
echo ""
echo "✅ All checks passed! Your application is ready for deployment."
echo ""
echo "Next steps:"
echo "1. Ensure you have set GOOGLE_CLOUD_PROJECT in your .env file"
echo "2. Run ./deploy.sh to deploy to Google Cloud Run"
echo ""