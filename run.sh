{
    set -e
    module purge
    module load machine_learning/2020.01-cudnn7.6-cuda10.1-py3.7
    source /sw/csi/anaconda3/4.4.0/binary/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate jax
    export TFDS_DATA_DIR="/encrypted/SFBData/liz0f/tensorflow_datasets"
    export exp_name="imagenet2012_debug_resume"
    source shell_scripts/resume-imagenet2012.sh
}
