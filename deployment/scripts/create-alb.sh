#!/bin/bash

set -e  # Exit on error

# Load environment variables
source deployment/scripts/load-env.sh
load_env
MAX_LENGTH=28
TRIMMED_APP_NAME="${APP_NAME:0:$MAX_LENGTH}"
TRIMMED_TG_NAME="${APP_NAME:0:$MAX_LENGTH}"

# Create ALB
echo "Creating Application Load Balancer..."
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name "${TRIMMED_APP_NAME}-alb" \
    --subnets "${VPC_SUBNET_1}" "${VPC_SUBNET_2}" \
    --security-groups "${SECURITY_GROUP}" \
    --scheme internet-facing \
    --type application \
    --region eu-north-1 \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

echo "ALB created with ARN: ${ALB_ARN}"

# Create Target Group
echo "Creating Target Group..."
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name "${TRIMMED_TG_NAME}-tg" \
    --protocol HTTP \
    --port 8000 \
    --vpc-id "${VPC_ID}" \
    --target-type ip \
    --health-check-path "/docs" \
    --health-check-interval-seconds 30 \
    --region eu-north-1 \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

echo "Target Group created with ARN: ${TARGET_GROUP_ARN}"

# Create Listener
echo "Creating Listener..."
LISTENER_ARN=$(aws elbv2 create-listener \
    --load-balancer-arn "${ALB_ARN}" \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn="${TARGET_GROUP_ARN}" \
    --region eu-north-1 \
    --query 'Listeners[0].ListenerArn' \
    --output text)

echo "Listener created with ARN: ${LISTENER_ARN}"

# Save ARNs to environment
cat > .env.tmp << EOF
$(cat .env)
TARGET_GROUP_ARN="${TARGET_GROUP_ARN}"
ALB_ARN="${ALB_ARN}"
LISTENER_ARN="${LISTENER_ARN}"
EOF

mv .env.tmp .env

echo "Load balancer setup complete!"