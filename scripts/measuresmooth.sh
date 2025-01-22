#!/bin/bash -l

#$ -N measureall           # Give job a name

module load miniconda
mamba activate stark

python3 python/measuresmooth.py
