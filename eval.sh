#!/bin/bash
#SBATCH --job-name=gemma3-270m-eval
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

set -euo pipefail

REPO=~/cs590llm/cs590-slm
LOGDIR=~/cs590llm/logs
mkdir -p "$LOGDIR"
cd "$REPO"
source venv/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
exec 1> >(tee "$LOGDIR/eval_${TIMESTAMP}.log")
exec 2>&1

NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l | xargs)}
NNODES=${SLURM_NNODES:-1}
echo ">>> Running on node: $(hostname)"
echo ">>> GPUs/node: $NGPUS | Nodes: $NNODES"

export NLTK_DATA="$HOME/nltk_data" # change this to your nltk_data directory for IFEval

python eval.py \
  --task arc-c \
  --model "$REPO/models/gemma270m-sft-v1" \
  --data_file "$REPO/data/arc_c_test.jsonl" \
  --data-size -1 \