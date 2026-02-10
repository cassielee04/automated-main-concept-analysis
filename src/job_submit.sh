#!/bin/bash
#$ -P skiran                    # Your project
#$ -l gpus=1                    # 1 GPU
#$ -l gpu_memory=32G            # GPU memory ≥32G
#$ -pe omp 4                    # 4 cores (OpenMP style)
#$ -l mem_per_core=6G           # 6G RAM per core
#$ -l h_rt=08:00:00             # 10-hour runtime limit
#$ -N transcribe_gpu            # Job name (optional)
#$ -j y                         # Merge stdout/stderr (optional, keeps output clean)

# Activate Conda env (your exact path)
source /projectnb/skiran/Cassie/Code/miniconda3/bin/activate transcribe
nvidia-smi > gpu_info.txt


# Your actual command here (runs on GPU node)
#python /projectnb/skiran/Cassie/mainconcpet_analysis/src/main.py
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python /projectnb/skiran/Cassie/mainconcpet_analysis/src/main.py

echo "Job completed at $(date)"