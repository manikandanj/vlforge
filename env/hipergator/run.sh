#!/bin/bash
#SBATCH --job-name=butterfly_project_setup
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
# To use multiple L4s for ~3x speedup, change above to: #SBATCH --gres=gpu:l4:3
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --output=/blue/arthur.porto-biocosmos/mjeyarajan3.gatech/butterfly_project/logs/run_%j.out


##Run the file as sbatch run.sh

SHARED_DIR="/blue/arthur.porto-biocosmos/share"
USER_DIR="/blue/arthur.porto-biocosmos/mjeyarajan3.gatech"
CONDA_ENV_NAME="butterfly_project"
CONDA_ENV_PATH=${SHARED_DIR}/conda/envs/${CONDA_ENV_NAME}
PROJECT_ROOT="${USER_DIR}/butterfly_project/code/vlforge"

#Exit immediately on error
set -euo pipefail

echo "Starting job at: $(date)"

# As recommended in https://docs.rc.ufl.edu/software/apps/conda/
if [ ! -d "${CONDA_ENV_PATH}" ]; then
    echo "Error: Conda environment not found at ${CONDA_ENV_PATH}"
    exit 1
fi
export PATH=${CONDA_ENV_PATH}/bin:$PATH

# Check if metadata file exists and get its size
METADATA_FILE="/blue/arthur.porto-biocosmos/data/datasets/nymphalidae_whole_specimen-v250613/metadata/data_meta-nymphalidae_whole_specimen-v250613.csv"
if [ -f "$METADATA_FILE" ]; then
    echo "Metadata file exists: $METADATA_FILE"
    echo "File size: $(du -h "$METADATA_FILE")"
    echo "Line count: $(wc -l < "$METADATA_FILE")"
else
    echo "WARNING: Metadata file not found: $METADATA_FILE"
    echo "Available files in metadata directory:"
    ls -la "/blue/arthur.porto-biocosmos/data/datasets/nymphalidae_whole_specimen-v250613/metadata/" || echo "Metadata directory not found"
fi

# Set optimal environment variables for performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Enable GPU optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.9"  # L4 architecture (Ada Lovelace)

echo "GPU Info:"
nvidia-smi
echo ""

echo "Running optimized embedding generation..."
python "${PROJECT_ROOT}/main.py" --config-name experiment_hipergator

echo "Job completed at: $(date)"