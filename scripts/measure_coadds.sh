#!/bin/bash -l

#$ -N lterv_coadds           # Give job a name

module load miniconda
mamba activate stark

python3 python/measurelte_coadds.py

