#!/bin/bash -l

#$ -N ltervs           # Give job a name

module load miniconda
mamba activate stark

python3 -m stark.measure lte
