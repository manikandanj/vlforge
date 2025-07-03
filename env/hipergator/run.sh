#!/bin/bash
#SBATCH --job-name=butterfly_project_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=1:00:00
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

# Verify GPU setup
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "Running main.py..."
python "${PROJECT_ROOT}/main.py"

echo "Job completed at: $(date)"