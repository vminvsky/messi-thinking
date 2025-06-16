#!/bin/bash
#SBATCH --job-name=generate_taco_base_llama_3.1_8b      # create a short name for your job
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
#SBATCH --array=0-15              # job array with index values 0, 1, 2, 3, 4

module purge
module load anaconda3/2024.6

export HF_HUB_OFFLINE=1
export HF_HOME="/scratch/gpfs/vv7118/models/"
export HF_TOKEN=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
export WANDB_MODE=offline
export WANDB_DISABLED=true
conda activate mech-int

vllm serve /scratch/gpfs/vv7118/models/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b --gpu-memory-utilization 0.8 --tensor-parallel-size 2 &
sleep 120
python single_model_inference.py \
    --model base \
    --base_model /scratch/gpfs/vv7118/models/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
    --begin_idx $(($SLURM_ARRAY_TASK_ID * 200)) \
    --output_dir /scratch/gpfs/vv7118/projects/messi-thinking/llama-3.1-8b/ \