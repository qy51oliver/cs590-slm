#!/bin/bash
#SBATCH --job-name=gemma3-270m-sft
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1 
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

set -euo pipefail

REPO=~/cs590llm/cs590-slm
LOGDIR=~/cs590llm/logs
mkdir -p "$LOGDIR"
cd "$REPO"

source venv/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
exec 1> >(tee ~/cs590llm/logs/ft_output_${TIMESTAMP}.log)
exec 2>&1

NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l | xargs)}
NNODES=${SLURM_NNODES:-1}

echo ">>> Running on node: $(hostname)"
echo ">>> GPUs/node: $NGPUS | Nodes: $NNODES"

torchrun --nproc_per_node="$NGPUS" finetune.py \
  --train_file data/sft_data.jsonl \
  --model google/gemma-3-270m \
  --output_dir models/gemma270m-sft-v1 \
  --fp16