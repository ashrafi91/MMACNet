#! /bin/bash
# Example SLURM submission script for training MMACNet on a shared GPU cluster.
# Adapt --partition/--gres/module names to your own cluster before use, and run
# this from the repo root (`sbatch scripts/job.slurm`).
# ====================================
#SBATCH --job-name=mmacnet_mimic3_full_all_codes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128GB
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h100
# ====================================

####### Set environment variables ###############
module load python/3.12.5
# Point these at your own credentialed MIMIC-III export; see datasets/README.md
export MIMIC_CSV_DIR=${MIMIC_CSV_DIR:-datasets/mimic3/csv}
export MIMIC_STATIC_DIR=${MIMIC_STATIC_DIR:-datasets/mimic3/static}

####### Run your script #########################
# python run_preprocessing.py --config_path configs/preprocessing/default/mimic3_full_all_codes.yml
python run.py --config_path configs/MMACNet/MMACNet_mimic3_full_all_codes.yml
python run.py --config_path configs/MMACNet/MMACNet_mimic3_full_all_codes.yml --test