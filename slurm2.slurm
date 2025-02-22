#!/bin/bash
#SBATCH --job-name=qwen_0.7      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --constraint=gpu80       # gpu type
#SBATCH --time=00:59:59          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=vv7118@princeton.edu
#SBATCH --array=0-10%1              # job array with index values 0, 1, 2, 3, 4

module purge
module load anaconda3/2024.6

MODEL="/scratch/gpfs/vv7118/models/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/b950d47742676362558ae821ef2202f847ac8109"
DIFFICULTY="MEDIUM"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DISABLED=true
export OPENAI_API_KEY=sk-proj-ubsVNO77hFXMyLfOPSAMT3BlbkFJ8ipc6bOO5ek4u9BPHY6d
conda activate mech-int

vllm serve $MODEL --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --port 1235 &
sleep 300 && \
conda activate skyteam && \
cd /scratch/gpfs/vv7118/projects/messi-thinking/SkyThought && \
python eval_taco.py --difficulty $DIFFICULTY --model $MODEL --port 1235