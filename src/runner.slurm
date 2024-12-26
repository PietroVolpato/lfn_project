#!/bin/bash

#SBATCH --job-name=AW_biological_128
#SBATCH --ntasks=1
#SBATCH --partition=allgroups
#SBATCH --time=02:00:00
#SBATCH --mem=30G
#SBATCH --output=embeddings_AW_biological_128_%j.csv
#SBATCH --error=AW_biological_128_%j.txt

# Carica Python dal modulo (se necessario)
module load python/3.x || true

# Trova Python e pip
export PYTHON=$(which python3 || which python)
export PIP="$PYTHON -m pip"

# Crea un ambiente virtuale temporaneo
$PYTHON -m venv /tmp/venv
source /tmp/venv/bin/activate

# Aggiorna pip
$PYTHON -m ensurepip --upgrade
$PIP install --upgrade pip

# Installa librerie richieste
$PIP install numpy pandas scipy

# Spostati nella cartella corretta
cd AttentionWalk

# Esegui lo script Python con srun
srun $PYTHON src/main.py --edge-path ../../data/citation/cit-HepTh_edges.csv --embedding-path ../../result/embeddings_citation_AW_128.csv