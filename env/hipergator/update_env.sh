#!/bin/bash
#SBATCH --job-name=butterfly_project_update
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --mem=8gb
#SBATCH --time=1:00:00

##Run the file as sbatch update_env.sh to update existing environment

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

# Check if environment exists
if [ ! -d "${CONDA_ENV_PATH}" ]; then
    echo "Error: Environment ${CONDA_ENV_NAME} not found at ${CONDA_ENV_PATH}"
    echo "Please run setup.sh first to create the environment"
    exit 1
fi

echo "Updating existing environment at $(date)"
echo "Environment path: ${CONDA_ENV_PATH}"
echo "Config file: ${CONDA_CONFIG_FILE}"

# Update environment with new dependencies
conda env update -f ${CONDA_CONFIG_FILE} -p ${CONDA_ENV_PATH} --prune -v

echo "Environment update completed at $(date)"

# Activate and verify
conda activate ${CONDA_ENV_PATH}

# Verify GPU setup with updated PyTorch
echo "Verifying updated environment:"
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version (PyTorch): {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    try:
        x = torch.randn(10, device='cuda')
        print('CUDA test: SUCCESS')
    except Exception as e:
        print(f'CUDA test: FAILED - {e}')
else:
    print('CUDA test: Not available')
"

conda deactivate

echo "Update process completed successfully!" 