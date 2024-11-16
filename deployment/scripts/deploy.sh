#deploy.sh
#!/bin/bash

set -e  # Exit on error

# Load environment variables
source deployment/scripts/load-env.sh
load_env

# Generate task definition from template
envsubst < deployment/ecs/task-definition.template.json > deployment/ecs/task-definition.json

# Register new task definition
echo "Registering new task definition..."
TASK_DEFINITION_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://deployment/ecs/task-definition.json \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

echo "Task definition registered: $TASK_DEFINITION_ARN"

# Update service
echo "Updating service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --task-definition $TASK_DEFINITION_ARN \
    --force-new-deployment

# Wait for deployment to complete
echo "Waiting for service to stabilize..."
aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME

# Check for any failed tasks in the last 5 minutes
FAILED_TASKS=$(aws ecs list-tasks \
    --cluster $CLUSTER_NAME \
    --desired-status STOPPED \
    --started-by $SERVICE_NAME \
    --since $(date -v-5M -u +%s) \
    --query 'taskArns[]' \
    --output text)

if [ ! -z "$FAILED_TASKS" ]; then
    echo "Found failed tasks. Fetching details..."
    aws ecs describe-tasks \
        --cluster $CLUSTER_NAME \
        --tasks $FAILED_TASKS
fi

echo "Deployment complete!"