#!/bin/bash

set -e  # Exit on error

# Load environment variables
source deployment/scripts/load-env.sh
load_env

echo "Creating VPC infrastructure for ${METRIC_APP_NAME}..."

# Create VPC
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --region ${AWS_REGION} \
    --query 'Vpc.VpcId' \
    --output text)

# Enable DNS hostnames
aws ec2 modify-vpc-attribute \
    --vpc-id $VPC_ID \
    --enable-dns-hostnames "{\"Value\":true}"

# Tag VPC
aws ec2 create-tags \
    --resources $VPC_ID \
    --tags Key=Name,Value=${METRIC_APP_NAME}-vpc

# Create Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
    --region ${AWS_REGION} \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)

aws ec2 attach-internet-gateway \
    --vpc-id $VPC_ID \
    --internet-gateway-id $IGW_ID

# Create subnets in different AZs
SUBNET1_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.1.0/24 \
    --availability-zone ${AWS_REGION}a \
    --query 'Subnet.SubnetId' \
    --output text)

SUBNET2_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.2.0/24 \
    --availability-zone ${AWS_REGION}b \
    --query 'Subnet.SubnetId' \
    --output text)

# Create route table
ROUTE_TABLE_ID=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --query 'RouteTable.RouteTableId' \
    --output text)

# Add route to Internet Gateway
aws ec2 create-route \
    --route-table-id $ROUTE_TABLE_ID \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID

# Associate route table with subnets
aws ec2 associate-route-table \
    --subnet-id $SUBNET1_ID \
    --route-table-id $ROUTE_TABLE_ID

aws ec2 associate-route-table \
    --subnet-id $SUBNET2_ID \
    --route-table-id $ROUTE_TABLE_ID

# Create security group
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
    --group-name "${METRIC_APP_NAME}-sg" \
    --description "Security group for ${METRIC_APP_NAME}" \
    --vpc-id $VPC_ID \
    --query 'GroupId' \
    --output text)

# Add security group rules
aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id $SECURITY_GROUP_ID \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

# Update .env file with new values
cat >> .env << EOF

# VPC Configuration
VPC_ID=${VPC_ID}
VPC_SUBNET_1=${SUBNET1_ID}
VPC_SUBNET_2=${SUBNET2_ID}
SECURITY_GROUP=${SECURITY_GROUP_ID}
EOF

echo "VPC setup complete!"