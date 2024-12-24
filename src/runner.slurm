#!/bin/bash

#SBATCH -J AW_biological_128
#SBATCH -n 1
#SBATCH -p allgroups
#SBATCH -t 02:00
#SBATCH -m 30G
#SBATCH -o embeddings_AW_biological_128_%j.csv
#SBATCH -e AW_biological_128_%j.txt

cd AttentionWalk 
srun python src/main.py --edge-path ../../data/biological/bio-CE-CX_edges.csv --embedding-path ../../result/embeddings_biological_AW_128.csv --dimensions 128