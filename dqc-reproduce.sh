#!/bin/bash

#SBATCH --array=1-2%16
#SBATCH --comment=dqc-reproduce

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=2

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))

declare -a commands=(
  [1]='MUJOCO_GL=egl python main.py --run_group=dqc-reproduce --offline_steps=1000000 --eval_interval=250000 --seed=100001 --agent=agents/dqc.py --agent.num_qs=2 --agent.policy_chunk_size=1 --agent.backup_horizon=25 --agent.use_chunk_critic=False --agent.distill_method=expectile --agent.implicit_backup_type=quantile --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent.q_agg=mean --agent.kappa_b=0.5 --tags="NS,n=25"'
  [2]='MUJOCO_GL=egl python main.py --run_group=dqc-reproduce --offline_steps=1000000 --eval_interval=250000 --seed=200002 --agent=agents/dqc.py --agent.num_qs=2 --agent.policy_chunk_size=1 --agent.backup_horizon=25 --agent.use_chunk_critic=False --agent.distill_method=expectile --agent.implicit_backup_type=quantile --env_name=humanoidmaze-giant-navigate-oraclerep-v0 --agent.q_agg=mean --agent.kappa_b=0.5 --tags="NS,n=25"'
)

parallel --delay 20 --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
        