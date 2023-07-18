#!/bin/bash

#SBATCH -t 06:00:00
#SBATCH -c 4
#SBATCH --job-name=Robert_example.csv
#SBATCH --mem=8GB

# This script was run in CESGA

##################################################
################    MODULES     ##################
##################################################

module load cesga/2020 miniconda3/4.11.0
module load intel/2021.3.0 crest/2.11.2
conda activate cheminf

##################################################
###############    MAIN SCRIPT     ###############
##################################################

python -m robert --ignore "[Name]" --names Name --y Target_values --csv_name Robert_example.csv