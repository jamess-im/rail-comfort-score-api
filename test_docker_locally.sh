#!/bin/bash
set -e

echo "=== Testing Docker Build and Run Locally ==="

# Build the Docker image
echo "Building Docker image..."
docker build -t train-comfort-api-test .

# Run the container with PORT environment variable
echo "Running container..."
docker run -d --name test-api -p 8080:8080 -e PORT=8080 train-comfort-api-test

# Wait for container to start
echo "Waiting for API to start..."
sleep 10

# Test the health endpoint
echo "Testing health endpoint..."
if curl -f http://localhost:8080/health; then
    echo -e "\n✅ Health check passed!"
else
    echo -e "\n❌ Health check failed!"
    echo "Container logs:"
    docker logs test-api
fi

# Clean up
echo -e "\nCleaning up..."
docker stop test-api
docker rm test-api

echo "Test complete!"