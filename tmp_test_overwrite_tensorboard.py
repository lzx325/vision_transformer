import os
import sys
from os.path import join,basename,dirname,splitext
from pathlib import Path
import numpy as np
import scipy
from scipy import io
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint
import argparse
from collections import defaultdict,OrderedDict,deque
import gzip
import shutil

from clu import metric_writers
if __name__=="__main__":
    # os.makedirs("tensorboard_debug_dir",exist_ok=True)
    # tensorboard_dir="train_dir/imagenet2012_pretrain_2/2021-06-13-13:00:27/tensorboard"
    # if not os.path.isdir("tensorboard_debug_dir/tensorboard_old"):
    #     shutil.copytree(tensorboard_dir,"tensorboard_debug_dir/tensorboard_old")
    
    # new_tensorboard_dir="tensorboard_debug_dir/tensorboard_new"
    # if os.path.isdir(new_tensorboard_dir):
    #     shutil.rmtree(new_tensorboard_dir)
    # shutil.copytree(tensorboard_dir,new_tensorboard_dir)
    # # os.rename("tensorboard_debug_dir/tensorboard","tensorboard_debug_dir/tensorboard_new")

    # writer = metric_writers.create_default_writer(new_tensorboard_dir, asynchronous=False)
    # for step in range(30000,44000,10):
    #     writer.write_scalars(step, dict(train_loss=5.0))
    # writer.close()

    csv_data_dir="tensorboard_debug_dir/csv_data"
    tb_from_csv_dir="tensorboard_debug_dir/tb_from_csv"
    if os.path.isdir(tb_from_csv_dir):
        shutil.rmtree(tb_from_csv_dir)
    os.makedirs(tb_from_csv_dir,exist_ok=True)
    writer=metric_writers.create_default_writer(tb_from_csv_dir, asynchronous=False)

    for csv_fp in Path(csv_data_dir).glob("*.csv"):
        csv_id=csv_fp.stem
        scalar_name=csv_id.split('-')[-1]
        print(scalar_name)
        scalar_df=pd.read_csv(csv_fp)
        scalar_df["Step"]=scalar_df["Step"].astype(np.int64)
        for index,row in scalar_df.iterrows():
            writer.write_scalars(row["Step"],{scalar_name:row["Value"]})
            
    writer.close()