{
    set -e
    source "/home/zhongxiao_li_kaust_edu_sa/anaconda3/etc/profile.d/conda.sh" 
    conda deactivate
    conda activate jax
    export TFDS_DATA_DIR="/data/liz0f/tensorflow_datasets"
    export exp_name="imagenet2012_resume"
    source shell_scripts/resume-imagenet2012_pretrain_2.sh
}
