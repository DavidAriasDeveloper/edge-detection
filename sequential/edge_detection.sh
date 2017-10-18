#!/bin/bash

#SBATCH --job-name=edge_detection
#SBATCH --output=res_edge_detection
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

./edge_detection felicity_ultra.jpg
