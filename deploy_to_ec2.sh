#!/bin/bash

# AWS EC2 Deployment Script for Hotel Booking Cancellation Prediction App

# Set variables
EC2_HOST="your-ec2-host.amazonaws.com"
EC2_USER="ec2-user"
EC2_KEY_PATH="~/.ssh/your-ec2-key.pem"
DOCKER_IMAGE="hotel-booking-cancellation"
DOCKER_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting deployment to AWS EC2...${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}AWS CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install it first.${NC}"
    exit 1
fi

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .

# Tag image for ECR (if using AWS ECR)
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
# docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} your-account.dkr.ecr.us-east-1.amazonaws.com/${DOCKER_IMAGE}:${DOCKER_TAG}

# Push to ECR (if using AWS ECR)
# docker push your-account.dkr.ecr.us-east-1.amazonaws.com/${DOCKER_IMAGE}:${DOCKER_TAG}

# Deploy to EC2 instance
echo -e "${YELLOW}Deploying to EC2 instance...${NC}"

# Copy docker-compose.yml to EC2
scp -i ${EC2_KEY_PATH} docker-compose.yml ${EC2_USER}@${EC2_HOST}:/home/${EC2_USER}/

# SSH into EC2 and run deployment
ssh -i ${EC2_KEY_PATH} ${EC2_USER}@${EC2_HOST} << 'EOF'
    # Update system
    sudo yum update -y
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        sudo amazon-linux-extras install docker -y
        sudo service docker start
        sudo usermod -a -G docker ec2-user
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    # Pull latest Docker image
    # docker pull your-account.dkr.ecr.us-east-1.amazonaws.com/${DOCKER_IMAGE}:${DOCKER_TAG}
    
    # Stop existing containers
    docker-compose down || true
    
    # Start new containers
    docker-compose up -d
    
    # Check if containers are running
    docker-compose ps
EOF

echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${YELLOW}Access your application at: http://${EC2_HOST}:8501${NC}"