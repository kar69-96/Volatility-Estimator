# Lambda Labs GPU Instance Setup Instructions

## Prerequisites
✅ You are already authenticated with Lambda Labs (`lambda-cli whoami` shows authenticated)
❌ SSH key needs to be added to Lambda Labs

## Step 1: Add SSH Key to Lambda Labs

Since `lambda-cli` shows 0 SSH keys, you need to add your SSH key through the Lambda Labs web interface:

1. **Go to Lambda Labs Dashboard**: https://cloud.lambdalabs.com/
2. **Navigate to SSH Keys section** (usually in Settings or Account settings)
3. **Click "Add SSH Key"** or similar button
4. **Copy your public key**:
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```
5. **Paste the key** into the Lambda Labs interface and give it a name (e.g., "my-key")
6. **Save the SSH key**

Alternatively, if the interactive launch handles SSH key setup, you can try:
```bash
cd /Users/karthikreddy/Downloads/GitHub/Demos/Volatility-Estimator
source venv310/bin/activate
lambda-cli interactive-launch
```

## Step 2: Launch GPU Instance

Once your SSH key is added, launch a 1x A10 GPU instance:

```bash
cd /Users/karthikreddy/Downloads/GitHub/Demos/Volatility-Estimator
source venv310/bin/activate

lambda-cli launch-instance \
  --region-name us-west-1 \
  --instance-type gpu_1x_a10 \
  --ssh-key-name <your-ssh-key-name> \
  --name volatility-training \
  --yes
```

**Replace `<your-ssh-key-name>`** with the name you gave your SSH key in Step 1.

**Note**: If you're unsure of the exact region or want to see available options, you can try:
```bash
lambda-cli interactive-launch
```

This will guide you through the process interactively.

## Step 3: Wait for Instance to Start

Check instance status:
```bash
lambda-cli list-instances
```

Wait until you see the instance with status "running" and note the IP address.

## Step 4: SSH Into the Instance

Once the instance is running, SSH into it:
```bash
ssh ubuntu@<instance-ip>
```

Replace `<instance-ip>` with the IP address from Step 3.

## Step 5: Setup Instance (After SSH)

Once you're SSH'd into the Lambda instance, run these commands:

### 5.1: Update System Packages
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 5.2: Install Python and Basic Tools
```bash
sudo apt-get install -y python3 python3-pip git
```

### 5.3: Clone or Copy Your Code

**Option A: If you have your code in a git repository:**
```bash
git clone <your-repo-url>
cd Volatility-Estimator
```

**Option B: Copy code from your local machine (from a new terminal on your Mac):**
```bash
# From your local machine, in the project directory:
cd /Users/karthikreddy/Downloads/GitHub/Demos/Volatility-Estimator
scp -r . ubuntu@<instance-ip>:~/volatility-estimator/
```

Then on the instance:
```bash
cd ~/volatility-estimator
```

### 5.4: Install Dependencies

```bash
# Install PyTorch with CUDA support (Lambda instances have CUDA)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip3 install transformers>=4.35.0 peft>=0.6.0 pandas>=2.0.0 numpy>=1.24.0 pyyaml>=6.0 scikit-learn>=1.3.0 arch>=6.2.0 tqdm

# Or if you copied the requirements.txt:
pip3 install -r requirements.txt
```

### 5.5: Verify GPU is Available
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

You should see:
```
CUDA available: True
GPU: NVIDIA A10
```

### 5.6: Run Training

```bash
# Make sure you're in the project directory
cd ~/volatility-estimator

# Create checkpoints directory
mkdir -p models/checkpoints

# Run training
python3 scripts/train.py
```

## Step 6: Download Trained Model (After Training Completes)

From your local machine:

```bash
cd /Users/karthikreddy/Downloads/GitHub/Demos/Volatility-Estimator
mkdir -p models/checkpoints

# Download the trained model
scp ubuntu@<instance-ip>:~/volatility-estimator/models/checkpoints/chronos.pt ./models/checkpoints/
```

## Step 7: Terminate Instance (When Done)

To save costs, terminate the instance when you're done:

```bash
cd /Users/karthikreddy/Downloads/GitHub/Demos/Volatility-Estimator
source venv310/bin/activate

# List instances to get the instance ID
lambda-cli list-instances

# Terminate the instance
lambda-cli terminate-instance <instance-id>
```

## Troubleshooting

**"SSH key not found" error:**
- Make sure you added the SSH key through the Lambda Labs web interface
- Use the exact name you gave the key (case-sensitive)

**"No capacity available" error:**
- Try a different region: `--region-name us-east-1` or `--region-name us-west-2`
- Wait a few minutes and try again
- Check Lambda Labs status page for availability

**Connection refused when SSHing:**
- Wait 1-2 minutes after launch for the instance to fully boot
- Verify the instance status shows "running"
- Double-check the IP address

**GPU not detected:**
- Lambda instances come with CUDA pre-installed, but you may need to use the correct PyTorch CUDA version
- Try: `pip3 install torch --index-url https://download.pytorch.org/whl/cu118`

