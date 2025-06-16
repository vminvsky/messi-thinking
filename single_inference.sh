#!/bin/bash
NODE_NAME=della-l08g4
CLUSTER_NUMBER=2

# Start deepseek server in a detached screen session
screen -dmS ${CLUSTER_NUMBER}_v2 ssh -t $NODE_NAME "cd /scratch/gpfs/vv7118/projects/messi-thinking && \
export HF_HUB_OFFLINE=1 && \
export HF_HOME="/scratch/gpfs/bs6865/reasoning-agents/_cache" && \
conda activate vllm && \
vllm serve /scratch/gpfs/bs6865/mergekit/s1_merge_0.7 --gpu_memory_utilization 0.8 --tensor_parallel_size 4 --port 8000" && \
sleep 240 && \
cd /scratch/gpfs/bs6865/messi-thinking/SkyThought && \
python eval_taco.py 
