#!/bin/bash

#git clone https://github.com/PietroVolpato/lfn_project.git

#SBATCH --job-name AW_biological_128
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --time 02:00
#SBATCH --mem 30G
#SBATCH --output embeddings_AW_biological_128_%j.csv
#SBATCH --error AW_biological_128_%j.txt

cd AttentionWalk 
srun python src/main.py --edge-path ../../data/proteins/PROTEINS-full.csv --embedding-path ../../result/embeddings_proteins_AW_128.csv