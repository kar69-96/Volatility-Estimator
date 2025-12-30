#!/bin/bash
# Install Lambda Guest Agent for monitoring
# This script installs the Guest Agent which collects system metrics (GPU, VRAM utilization)
# and sends them to Lambda's backend for viewing in the Lambda Cloud console.

set -e

echo "=========================================="
echo "Lambda Guest Agent Installation"
echo "=========================================="
echo ""
echo "The Guest Agent collects system metrics (GPU, VRAM utilization)"
echo "and sends them to Lambda's backend for monitoring."
echo ""

# Check if already installed
if systemctl list-units --type=service --state=running | grep -q "lambda-guest-agent"; then
    echo "Guest Agent appears to be already installed and running."
    echo ""
    read -p "Do you want to reinstall? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping installation."
        exit 0
    fi
fi

# Download and install the Guest Agent
echo ""
echo "Step 1: Downloading and installing Guest Agent..."
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

# Wait a moment for the service to start
sleep 3

# Verify installation
echo ""
echo "Step 2: Verifying Guest Agent installation..."
if sudo systemctl --no-pager status lambda-guest-agent.service > /dev/null 2>&1; then
    echo ""
    echo "✓ Guest Agent service is running"
    echo ""
    echo "Service status:"
    sudo systemctl --no-pager status lambda-guest-agent.service | head -n 10
    echo ""
    echo "✓ Guest Agent updater timer is active"
    sudo systemctl --no-pager status lambda-guest-agent-updater.timer | head -n 5
else
    echo "⚠ Warning: Guest Agent service may not be running properly"
    echo "Checking status..."
    sudo systemctl --no-pager status lambda-guest-agent.service || true
fi

echo ""
echo "=========================================="
echo "✓ Guest Agent installation complete!"
echo "=========================================="
echo ""
echo "Your metrics should appear in the Lambda Cloud console within a few minutes."
echo ""
echo "Note: The Guest Agent automatically updates itself every two weeks."
echo ""
echo "To check status manually:"
echo "  sudo systemctl status lambda-guest-agent.service"
echo ""
echo "To disable automatic updates:"
echo "  sudo systemctl stop lambda-guest-agent-updater.timer"
echo "  sudo systemctl disable lambda-guest-agent-updater.timer"
echo ""
echo "To uninstall:"
echo "  sudo apt remove lambda-guest-agent"
echo ""

