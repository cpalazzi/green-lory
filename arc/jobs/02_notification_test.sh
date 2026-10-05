#!/bin/bash
#SBATCH --job-name=green-lory-mail-test
#SBATCH --partition=short
#SBATCH --clusters=htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128M
#SBATCH --time=00:02:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=carlo.palazzi@eng.ox.ac.uk
set -euo pipefail
date -u
echo "Green Lory notification test: job ${SLURM_JOB_ID} on ${SLURM_CLUSTER_NAME}"
scontrol --oneliner show job "$SLURM_JOB_ID"
sleep 15
echo "Notification test completed. Scheduler settings do not prove mailbox delivery."
