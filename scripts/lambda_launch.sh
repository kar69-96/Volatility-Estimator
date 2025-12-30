#!/bin/bash
# Quick launch script for Lambda Labs A100 instance
# Usage: ./scripts/lambda_launch.sh [instance-type]

set -e

INSTANCE_TYPE=${1:-gpu_1x_a100}
SSH_KEY_NAME=${LAMBDA_SSH_KEY_NAME:-lambda-key}
REGION=${LAMBDA_REGION:-us-west-1}

echo "=========================================="
echo "Lambda Labs A100 Launch Script"
echo "=========================================="
echo ""
echo "Instance Type: $INSTANCE_TYPE"
echo "SSH Key: $SSH_KEY_NAME"
echo "Region: $REGION"
echo ""

# Check if Lambda CLI is installed
if ! command -v lambdacloud &> /dev/null; then
    echo "Error: Lambda CLI not found. Install with: pip install lambdacloud"
    exit 1
fi

# Check if authenticated
if ! lambdacloud auth status &> /dev/null; then
    echo "Warning: Not authenticated. Run: lambdacloud auth login"
    echo "Or set LAMBDA_API_KEY environment variable"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Launch instance
echo "Launching instance..."
INSTANCE_OUTPUT=$(lambdacloud instance launch \
    --instance-type "$INSTANCE_TYPE" \
    --ssh-key-name "$SSH_KEY_NAME" \
    --region "$REGION" \
    --format json)

INSTANCE_ID=$(echo "$INSTANCE_OUTPUT" | jq -r '.id')
INSTANCE_IP=$(echo "$INSTANCE_OUTPUT" | jq -r '.ip')

if [ "$INSTANCE_ID" == "null" ] || [ -z "$INSTANCE_ID" ]; then
    echo "Error: Failed to launch instance"
    echo "$INSTANCE_OUTPUT"
    exit 1
fi

echo ""
echo "✓ Instance launched successfully!"
echo "  Instance ID: $INSTANCE_ID"
echo "  IP Address: $INSTANCE_IP"
echo ""
echo "Waiting for instance to be ready (this may take 1-2 minutes)..."
echo ""

# Wait for instance to be ready
MAX_WAIT=300  # 5 minutes
WAIT_TIME=0
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    STATUS=$(lambdacloud instance get "$INSTANCE_ID" --format json | jq -r '.status')
    if [ "$STATUS" == "running" ]; then
        echo "✓ Instance is ready!"
        break
    fi
    echo "  Status: $STATUS (waiting...)"
    sleep 10
    WAIT_TIME=$((WAIT_TIME + 10))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "Warning: Instance took too long to start. Please check manually."
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. SSH into the instance:"
echo "   lambdacloud instance ssh $INSTANCE_ID"
echo ""
echo "2. Or use standard SSH:"
echo "   ssh -i ~/.ssh/lambda_key ubuntu@$INSTANCE_IP"
echo ""
echo "3. Once connected, set up the project:"
echo "   git clone <your-repo-url>"
echo "   cd Volatility-Estimator"
echo "   bash scripts/setup_lambda.sh"
echo ""
echo "4. Run training:"
echo "   source venv/bin/activate"
echo "   python3 scripts/train_sample.py"
echo ""
echo "5. When done, terminate the instance:"
echo "   lambdacloud instance terminate $INSTANCE_ID"
echo ""
echo "=========================================="
echo "Instance Details:"
echo "=========================================="
echo "Instance ID: $INSTANCE_ID"
echo "IP Address: $INSTANCE_IP"
echo ""
echo "Save this information to terminate the instance later!"
echo ""

