mkdir -p slurm
sbatch <<- EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -J ViT
#SBATCH -o slurm/ViT.%J.out
#SBATCH -e slurm/ViT.%J.err
#SBATCH --time=144:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --constraint="[v100]"
#run the application:
bash run.sh
}
EOF
