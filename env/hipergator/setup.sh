#!/bin/bash
#SBATCH --job-name=butterfly_project_setup
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=1:00:00

##Run the file as sbatch setup.sh after placing environment.yml file

SHARED_DIR="/blue/arthur.porto-biocosmos/share"
USER_DIR="/blue/arthur.porto-biocosmos/mjeyarajan3.gatech"
CONDA_ENV_NAME="butterfly_project"
CONDA_CONFIG_FILE="${USER_DIR}/butterfly_project/code/vlforge/environment.yml"
CONDA_ENV_PATH=${SHARED_DIR}/conda/envs/${CONDA_ENV_NAME}

#Exit immediately on error
set -euo pipefail

module load conda

#Initialize conda for this shell session
eval "$(conda shell.bash hook)"

# activate conda only for installing packages
echo "Starting environment creation at $(date)"
conda env create -f ${CONDA_CONFIG_FILE} -p ${CONDA_ENV_PATH} -v -y
echo "Environment creation completed at $(date)"
conda activate ${CONDA_ENV_PATH}

# Verify GPU setup
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

conda deactivate
