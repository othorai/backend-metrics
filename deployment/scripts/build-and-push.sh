#!/bin/bash

set -e  # Exit on error

# Load environment variables
source deployment/scripts/load-env.sh
load_env

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com || handle_error "Failed to login to ECR"

# Build and push the image directly
echo "Building and pushing Docker image..."
docker buildx build \
    --platform linux/amd64 \
    --push \
    --build-arg DB_HOST=$DB_HOST \
    --build-arg DB_PORT=$DB_PORT \
    --build-arg DB_NAME=$DB_NAME \
    -t $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME:latest \
    -f deployment/Dockerfile . || handle_error "Failed to build and push image"

echo "Build and push complete!"