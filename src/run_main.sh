#!/bin/bash
#$ -P skiran
#$ -l gpus=1
#$ -l gpu_memory=32G
#$ -pe omp 4
#$ -l mem_per_core=6G
#$ -l h_rt=08:00:00
#$ -j y
#$ -o logs/mainconcept.$JOB_NAME.$JOB_ID.log

source /projectnb/skiran/Cassie/Code/miniconda3/bin/activate transcribe
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

COHORT="$1"
python /projectnb/skiran/Cassie/mainconcpet_analysis/src/main.py --cohort "$COHORT"
