# Lambda Labs Setup Guide

Complete guide to set up and run training on Lambda Labs using the Lambda CLI.

## Prerequisites

1. **Lambda Labs Account**: Sign up at [lambdalabs.com](https://lambdalabs.com)
2. **Lambda CLI**: Install the Lambda CLI tool
3. **SSH Key**: Generate an SSH key pair for instance access

## Step 1: Install Lambda CLI

You can use either CLI tool for managing Lambda Labs instances:

**Option 1: Official Lambda Labs CLI (Recommended)**
```bash
pip install lambdacloud
```

**Option 2: Third-party Lambda CLI**
```bash
pip install lambda-cli
```

**Note**: The training scripts support both CLIs. The official `lambdacloud` is recommended, but `lambda-cli` works as an alternative. Both will be installed automatically by the setup script.

### Add to PATH (if command not found)

If `lambdacloud` command is not found after installation, add Python's bin directory to your PATH:

```bash
# For macOS/Linux - add to ~/.zshrc or ~/.bashrc
export PATH="$HOME/Library/Python/3.9/bin:$PATH"

# Then reload your shell
source ~/.zshrc  # or source ~/.bashrc

# Verify installation
lambdacloud --version
```

Alternatively, you can use the full path:
```bash
# Use full path to the command
~/Library/Python/3.9/bin/lambdacloud --version
```

## Step 2: Authenticate

```bash
# Login to Lambda Labs (you'll need your API key from lambdalabs.com)
lambdacloud auth login

# Or set API key as environment variable
export LAMBDA_API_KEY="your-api-key-here"
```

## Step 3: Set Up SSH Key

```bash
# Generate SSH key if you don't have one
ssh-keygen -t rsa -b 4096 -C "your-email@example.com" -f ~/.ssh/lambda_key

# Add the public key to Lambda Labs
lambdacloud ssh-key add ~/.ssh/lambda_key.pub --name lambda-key

# List your SSH keys to verify
lambdacloud ssh-key list
```

## Step 4: Launch A100 Instance

### Filesystem Configuration

When launching an instance, Lambda Labs will ask about filesystem configuration. Here's what to choose:

**Option 1: No Persistent Storage (Recommended for one-time training)**
- Choose: **"Proceed without persistent storage"** or **"No filesystem"**
- **When to use**: Single training run, small datasets, model will be downloaded after training
- **Pros**: No additional storage costs, simpler setup
- **Cons**: Data is lost when instance terminates (but model is saved and downloaded)

**Option 2: Create New Filesystem (For multiple training runs)**
- Choose: **"Create a new filesystem"**
- **When to use**: 
  - Multiple training runs on the same dataset
  - Large datasets that take time to download
  - Want to keep checkpoints across sessions
- **Size recommendation**: 50-100GB (enough for datasets + models)
- **Pros**: Data persists across instance terminations
- **Cons**: Additional cost (~$0.10/GB/month), must remember to delete filesystem

**Option 3: Attach Existing Filesystem**
- Choose: **"Attach existing filesystem"**
- **When to use**: You already have a filesystem with data/models
- **Note**: Filesystem and instance must be in the same region

### Recommendation

For this training script:
- **Use Option 1 (No persistent storage)** if:
  - This is a one-time training run
  - You'll download the model checkpoint after training
  - Dataset is small (<10GB) and quick to download
  
- **Use Option 2 (Create new filesystem)** if:
  - You plan multiple training runs
  - Dataset is large and takes time to download
  - You want to keep intermediate checkpoints

### Launch Commands

**Using Official CLI (`lambdacloud`):**

```bash
# Launch a single A100 instance (40GB)
lambdacloud instance launch \
  --instance-type gpu_1x_a100 \
  --ssh-key-name lambda-key \
  --region us-west-1

# Alternative: Launch A100 80GB
lambdacloud instance launch \
  --instance-type gpu_1x_a100_80gb \
  --ssh-key-name lambda-key \
  --region us-west-1

# Cheaper alternative: A10
lambdacloud instance launch \
  --instance-type gpu_1x_a10 \
  --ssh-key-name lambda-key \
  --region us-west-1
```

**Using Third-party CLI (`lambda-cli`):**

```bash
# First, authenticate (if not already done)
lambda auth

# Launch a single A100 instance (40GB)
lambda up --instance_type=gpu.1x.a100

# Alternative: Launch A100 80GB
lambda up --instance_type=gpu.1x.a100.80gb

# Cheaper alternative: A10
lambda up --instance_type=gpu.1x.a10

# List running instances
lambda ls

# Get instance details (after launch)
lambda ls  # Shows instance IDs and status
```

**Note**: Instance type naming differs between CLIs:
- **lambdacloud**: `gpu_1x_a100`, `gpu_1x_a100_80gb`, `gpu_1x_a10`
- **lambda-cli**: `gpu.1x.a100`, `gpu.1x.a100.80gb`, `gpu.1x.a10`

**Via Web Dashboard:**
- Go to [lambdalabs.com](https://lambdalabs.com)
- When prompted for filesystem: Choose "No filesystem" for one-time training

### Important Notes

1. **Default Root Volume**: All instances come with a root volume (~200GB) that's sufficient for training. This is **ephemeral** (lost on termination).

2. **Persistent Storage Cost**: Filesystems are billed per GB/month (~$0.10/GB). Remember to delete unused filesystems!

3. **Region Matching**: Filesystem and instance must be in the same region.

4. **Auto-Termination**: Since the training script auto-terminates the instance, any data on the root volume will be lost. Make sure to:
   - Download model checkpoints before termination
   - Or use persistent storage if you need to keep data

## Step 5: Get Instance Details

**Using Official CLI (`lambdacloud`):**

```bash
# List all instances
lambdacloud instance list

# Get specific instance details (replace INSTANCE_ID)
lambdacloud instance get INSTANCE_ID

# Get SSH connection info
lambdacloud instance get INSTANCE_ID --format json | jq '.ssh_command'
```

**Using Third-party CLI (`lambda-cli`):**

```bash
# List all instances
lambda ls

# List with more details (if supported)
lambda ls --verbose

# Get instance details (replace INSTANCE_ID)
# Note: lambda-cli may have different commands for detailed info
lambda ls | grep INSTANCE_ID
```

## Step 6: Connect to Instance

**Using Official CLI (`lambdacloud`):**

```bash
# SSH into the instance (replace INSTANCE_ID)
lambdacloud instance ssh INSTANCE_ID
```

**Using Third-party CLI (`lambda-cli`):**

```bash
# lambda-cli doesn't have a direct SSH command
# Use standard SSH instead (get IP from 'lambda ls')
ssh -i ~/.ssh/lambda_key ubuntu@<instance-ip>
```

**Standard SSH (works with both CLIs):**

```bash
# Get instance IP from either CLI, then:
ssh -i ~/.ssh/lambda_key ubuntu@<instance-ip>
```

## Step 7: Set Up Project on Instance

Once connected to the Lambda instance:

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install git if not already installed
sudo apt-get install -y git

# Clone your repository (or upload via SCP)
git clone <your-repo-url>
cd Volatility-Estimator

# Or upload files via SCP from your local machine:
# scp -i ~/.ssh/lambda_key -r /path/to/Volatility-Estimator ubuntu@<instance-ip>:~/
```

## Step 8: Run Setup Script

On the Lambda instance:

```bash
# Make setup script executable
chmod +x scripts/setup_lambda.sh

# Run the setup script
bash scripts/setup_lambda.sh
```

The setup script will:
- Create a virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Verify GPU availability

## Step 8.5: Install Guest Agent (Optional but Recommended)

The Lambda Guest Agent collects system metrics (GPU, VRAM utilization) and sends them to Lambda's backend. You can view these metrics in the Lambda Cloud console.

**Note**: The metrics dashboard will be generally available in Q2 2025. To request early access, submit a support ticket.

To install the Guest Agent:

```bash
# Make the install script executable
chmod +x scripts/install_guest_agent.sh

# Run the installation script
bash scripts/install_guest_agent.sh
```

**Or install manually:**

```bash
# Download and install the Guest Agent
curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

# Verify it's running
sudo systemctl --no-pager status lambda-guest-agent.service
```

Your metrics should appear in the Lambda Cloud console within a few minutes.

**Additional Guest Agent Commands:**

```bash
# Check status
sudo systemctl status lambda-guest-agent.service

# Disable automatic updates (updates every 2 weeks by default)
sudo systemctl stop lambda-guest-agent-updater.timer
sudo systemctl disable lambda-guest-agent-updater.timer

# Uninstall (if needed)
sudo apt remove lambda-guest-agent
```

## Step 9: Run Training

```bash
# Activate virtual environment
source venv/bin/activate

# Run training (automatically detects A100 and optimizes)
python3 scripts/train_sample.py
```

The training script will automatically:
- Detect A100 GPU
- Use batch size 128 (optimized for A100)
- Enable mixed precision (FP16) training
- Use GPU 0 only (CUDA_VISIBLE_DEVICES=0)
- **Auto-terminate the instance when training completes** (saves costs!)

### Auto-Termination Feature

Both `train_sample.py` and `train_sp500.py` automatically terminate the Lambda instance when training completes. This prevents unnecessary charges.

**To disable auto-termination** (if you want to keep the instance running):
```bash
export LAMBDA_NO_AUTO_TERMINATE=1
python3 scripts/train_sample.py
```

**To manually set instance ID** (if auto-detection fails):
```bash
export LAMBDA_INSTANCE_ID="your-instance-id"
python3 scripts/train_sample.py
```

## Step 10: Monitor Training

Training will output progress like:
```
Epoch 1/50 | Train Loss: 0.1234 | Val Loss: 0.1456 | Val QLIKE: 0.2345
Epoch 2/50 | Train Loss: 0.1123 | Val Loss: 0.1345 | Val QLIKE: 0.2234
...
```

## Step 11: Download Results

After training completes, download the model checkpoint:

```bash
# From your local machine, download the trained model
scp -i ~/.ssh/lambda_key \
  ubuntu@<instance-ip>:~/Volatility-Estimator/models/checkpoints/chronos_5ticker.pt \
  ./models/checkpoints/
```

## Step 12: Stop Instance (Automatic!)

**The training scripts automatically terminate the instance when training completes!**

If you need to manually stop the instance:

**Using Official CLI (`lambdacloud`):**
```bash
lambdacloud instance terminate INSTANCE_ID
```

**Using Third-party CLI (`lambda-cli`):**
```bash
lambda rm INSTANCE_ID
```

**From within the instance (last resort):**
```bash
sudo shutdown -h now
```
**Note**: `sudo shutdown` only shuts down the OS but may not stop Lambda Labs billing. Use the CLI commands above to properly terminate.

**Note**: If auto-termination fails, the script will print instructions on how to manually terminate.

**Important**: The script uses the Lambda Labs CLI to properly terminate instances. It supports both:
- `lambdacloud` (official CLI): `lambdacloud instance terminate <id>`
- `lambda-cli` (third-party): `lambda rm <id>`

If neither CLI is installed or authenticated on the instance, auto-termination will fail. Make sure to:
1. Install at least one CLI on the instance: `pip install lambdacloud` OR `pip install lambda-cli`
2. Authenticate:
   - For `lambdacloud`: `lambdacloud auth login` or set `LAMBDA_API_KEY` environment variable
   - For `lambda-cli`: `lambda auth`
3. If auto-termination fails, manually terminate from the Lambda Labs dashboard to avoid continued billing

## Quick Reference Commands

**Official CLI (`lambdacloud`):**

```bash
# List instances
lambdacloud instance list

# Launch instance
lambdacloud instance launch --instance-type gpu_1x_a100 --ssh-key-name lambda-key

# SSH into instance
lambdacloud instance ssh INSTANCE_ID

# Stop instance
lambdacloud instance terminate INSTANCE_ID

# View instance status
lambdacloud instance get INSTANCE_ID
```

**Third-party CLI (`lambda-cli`):**

```bash
# Authenticate (first time)
lambda auth

# List instances
lambda ls

# Launch instance
lambda up --instance_type=gpu.1x.a100

# Stop instance
lambda rm INSTANCE_ID

# View instance status
lambda ls  # Shows all instances with status
```

## Troubleshooting

### Instance won't launch
- Check your account has available quota
- Verify SSH key is added correctly
- Try a different region

### GPU not detected
- Verify CUDA is installed: `nvidia-smi`
- Check PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Out of memory errors
- Reduce batch size in `train_sample.py` (change from 128 to 64)
- Or use gradient accumulation

### Connection issues
- Verify SSH key permissions: `chmod 600 ~/.ssh/lambda_key`
- Check instance is running: `lambdacloud instance list`

### Instance didn't auto-terminate after training
**Problem**: Training completed but instance is still running and billing.

**Causes**:
1. Lambda CLI (`lambdacloud`) not installed on the instance
2. Lambda CLI not authenticated (no API key)
3. Instance ID could not be auto-detected
4. Network issues preventing API call

**Solutions**:
1. **Install Lambda CLI on instance** (install at least one):
   ```bash
   # Option 1: Official CLI
   pip install lambdacloud
   lambdacloud auth login  # Or set LAMBDA_API_KEY env var
   
   # Option 2: Third-party CLI
   pip install lambda-cli
   lambda auth
   ```

2. **Manually set instance ID** before training:
   ```bash
   export LAMBDA_INSTANCE_ID="your-instance-id"
   python3 scripts/train_sample.py
   ```

3. **Manually terminate** from your local machine:
   ```bash
   # Using official CLI
   lambdacloud instance terminate <instance-id>
   
   # OR using third-party CLI
   lambda rm <instance-id>
   ```

4. **Check instance status**:
   ```bash
   lambdacloud instance list
   ```

**Note**: The old version of `train_sample.py` used `sudo shutdown -h now`, which only shuts down the OS but doesn't properly terminate the Lambda Labs instance. The updated version uses the Lambda Labs CLI (`lambdacloud instance terminate` or `lambda rm`) which properly stops billing.

## Cost Optimization

- **Stop instances immediately** after training completes
- Use `gpu_1x_a10` for testing (cheaper than A100)
- Monitor usage in Lambda Labs dashboard
- Set up billing alerts

## Example Complete Workflow

```bash
# 1. Launch instance
INSTANCE_ID=$(lambdacloud instance launch \
  --instance-type gpu_1x_a100 \
  --ssh-key-name lambda-key \
  --region us-west-1 \
  --format json | jq -r '.id')

# 2. Wait for instance to be ready
sleep 30

# 3. SSH and setup
lambdacloud instance ssh $INSTANCE_ID << 'EOF'
  git clone <your-repo-url>
  cd Volatility-Estimator
  bash scripts/setup_lambda.sh
  source venv/bin/activate
  python3 scripts/train_sample.py
EOF

# 4. Download results
scp -i ~/.ssh/lambda_key \
  ubuntu@$(lambdacloud instance get $INSTANCE_ID --format json | jq -r '.ip'):~/Volatility-Estimator/models/checkpoints/chronos_5ticker.pt \
  ./models/checkpoints/

# 5. Terminate instance
lambdacloud instance terminate $INSTANCE_ID
```

