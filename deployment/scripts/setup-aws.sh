#!/bin/bash

# Load environment variables
source deployment/scripts/load-env.sh
load_env

# Create ECR repository (if doesn't exist)
echo "Creating/verifying ECR repository..."
aws ecr describe-repositories --repository-names ${APP_NAME} 2>/dev/null || \
aws ecr create-repository --repository-name ${APP_NAME}

# Store/update secrets in Parameter Store
echo "Storing/updating secrets in Parameter Store..."
aws ssm put-parameter --name "/fastapi/db_user" --value "$DB_USER" --type SecureString --overwrite
aws ssm put-parameter --name "/fastapi/db_password" --value "$DB_PASSWORD" --type SecureString --overwrite
aws ssm put-parameter --name "/fastapi/secret_key" --value "$SECRET_KEY" --type SecureString --overwrite
aws ssm put-parameter --name "/fastapi/openai_api_key" --value "$OPENAI_API_KEY" --type SecureString --overwrite

# Service Linked Role - Skip if exists
echo "Verifying ECS Service Linked Role..."
aws iam get-role --role-name AWSServiceRoleForECS 2>/dev/null || \
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com

# Task Execution Role - Skip if exists
echo "Verifying ECS Task Execution Role..."
aws iam get-role --role-name ecsTaskExecutionRole 2>/dev/null || \
{
    echo "Creating ECS Task Execution Role..."
    aws iam create-role \
        --role-name ecsTaskExecutionRole \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "ecs-tasks.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }'

    # Attach necessary policies
    aws iam attach-role-policy \
        --role-name ecsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

    aws iam attach-role-policy \
        --role-name ecsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess
}

# Update Parameter Store access policy
echo "Updating Parameter Store access policy..."
aws iam put-role-policy \
    --role-name ecsTaskExecutionRole \
    --policy-name ParameterStoreAccess \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "ssm:GetParameters",
                    "ssm:GetParameter",
                    "ssm:GetParametersByPath"
                ],
                "Resource": "arn:aws:ssm:'$AWS_REGION':'$AWS_ACCOUNT_ID':parameter/fastapi/*"
            }
        ]
    }'

# Create or update ECS cluster
echo "Creating/updating ECS cluster..."
aws ecs create-cluster \
    --cluster-name $CLUSTER_NAME \
    --capacity-providers FARGATE \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
    --tags key=Environment,value=production

# Create or update CloudWatch log group
echo "Creating/updating CloudWatch log group..."
aws logs create-log-group --log-group-name "/ecs/${APP_NAME}" 2>/dev/null || true

echo "Setup complete!"