#!/bin/bash

#SBATCH --job-name=dqc-reproduce
#SBATCH --open-mode=append
#SBATCH -o /global/scratch/users/jenniferzhao/logs/%A_%a.out
#SBATCH -e /global/scratch/users/jenniferzhao/logs/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --requeue
#SBATCH --array=1-2%16
#SBATCH --comment=dqc-reproduce

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=2

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))

# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl

declare -a commands=(
  [1]='MUJOCO_GL=egl python main.py --run_group=dqc-reproduce --offline_steps=1000000 --eval_interval=250000 --eval_episodes=1 --seed=100001 --agent=agents/dqc.py --agent.num_qs=2 --agent.policy_chunk_size=5 --agent.backup_horizon=25 --agent.use_chunk_critic=True --agent.distill_method=expectile --agent.implicit_backup_type=quantile --env_name=cube-quadruple-play-oraclerep-v0 --agent.q_agg=min --agent.kappa_b=0.93 --agent.kappa_d=0.8 --tags="DQC,h=25,ha=5" --save_dir=../scratch/dqc-reproduce/ --dataset_dir ../scratch/data/cube-quadruple-play-v0/'
  [2]='MUJOCO_GL=egl python main.py --run_group=dqc-reproduce --offline_steps=1000000 --eval_interval=250000 --eval_episodes=1 --seed=200002 --agent=agents/dqc.py --agent.num_qs=2 --agent.policy_chunk_size=5 --agent.backup_horizon=25 --agent.use_chunk_critic=True --agent.distill_method=expectile --agent.implicit_backup_type=quantile --env_name=cube-quadruple-play-oraclerep-v0 --agent.q_agg=min --agent.kappa_b=0.93 --agent.kappa_d=0.8 --tags="DQC,h=25,ha=5" --save_dir=../scratch/dqc-reproduce/ --dataset_dir ../scratch/data/cube-quadruple-play-v0/'
)

parallel --delay 20 --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
        
