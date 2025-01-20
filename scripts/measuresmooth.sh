#!/bin/bash -l

#$ -N smooth          # Give job a name

module load miniconda
mamba activate stark

python3 python/measuresmooth.py
