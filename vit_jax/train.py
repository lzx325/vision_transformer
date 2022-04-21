# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import glob
import os
from os.path import join, splitext, basename
import ipdb
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import time
import yaml
from clu import metric_writers

import numpy as np
import pandas as pd
from pathlib import Path

import jax
import jax.numpy as jnp

import flax
import flax.optim as optim
import flax.jax_utils as flax_utils

import tensorflow as tf
from datetime import datetime as dt

from vit_jax import checkpoint
from vit_jax import flags
from vit_jax import hyper
from vit_jax import vit_logging
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import momentum_clip


def make_update_fn(vit_fn, accum_steps,weight_decay):

  # Update step, replicated over all TPUs/GPUs
  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
  def update_fn(opt, lr, batch, update_rng):

    # Bind the rng key to the device id (which is unique across hosts)
    # Note: This is only used for multi-host training (i.e. multiple computers
    # each with multiple accelerators).
    update_rng = jax.random.fold_in(update_rng, jax.lax.axis_index('batch'))
    update_rng, new_update_rng = jax.random.split(update_rng)

    def cross_entropy_loss(*, logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))

    def loss_fn(params, images, labels):
      with flax.nn.stochastic(update_rng): # lizx: for dropout
        logits = vit_fn(params, images, train=True)
      cls_loss=cross_entropy_loss(logits=logits, labels=labels)
      if weight_decay>0:
        weight_penalty_params = jax.tree_leaves(params)
        weight_l2_loss = sum([jnp.sum(x ** 2)
                        for x in weight_penalty_params
                        if x.ndim > 1])
        return cls_loss+0.5*weight_decay*weight_l2_loss
      else:
        return cls_loss

    l, g = hyper.accumulate_gradient(
        jax.value_and_grad(loss_fn), opt.target, batch['image'], batch['label'],
        accum_steps
    )
    g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g) # lizx: average gradients across devices

    opt = opt.apply_gradient(g, learning_rate=lr)
    return opt, l, new_update_rng

  return update_fn


def main(args):
  train_dir = os.path.join(args.train_dir, args.exp_name, dt.now().strftime('%Y-%m-%d-%H:%M:%S'))
  log_dir=os.path.join(train_dir,"log")
  tensorboard_dir=os.path.join(train_dir,"tensorboard")
  ckpt_dir=os.path.join(train_dir,"ckpt")
  config_dir=join(train_dir,"config")

  os.makedirs(log_dir,exist_ok=True)
  os.makedirs(tensorboard_dir,exist_ok=True)
  os.makedirs(ckpt_dir,exist_ok=True)
  os.makedirs(config_dir,exist_ok=True)
  write_args_to_yaml(args,join(config_dir,"config.yaml"))
  

  logger = vit_logging.setup_logger(log_dir)
  logger.info(args)

  logger.info(f'Available devices: {jax.devices()}')

  # Setup input pipeline
  dataset_info = input_pipeline.get_dataset_info(args.dataset, 'train')

  ds_train = input_pipeline.get_data(
      dataset=args.dataset,
      mode='train',
      repeats=None,
      mixup_alpha=args.mixup_alpha,
      batch_size=args.batch,
      shuffle_buffer=args.shuffle_buffer,
      tfds_data_dir=args.tfds_data_dir,
      tfds_manual_dir=args.tfds_manual_dir)
  batch = next(iter(ds_train))
  logger.info(ds_train)
  ds_test = input_pipeline.get_data(
      dataset=args.dataset,
      mode='test',
      repeats=1,
      batch_size=args.batch_eval,
      tfds_data_dir=args.tfds_data_dir,
      tfds_manual_dir=args.tfds_manual_dir)
  logger.info(ds_test)

  # Build VisionTransformer architecture
  model = models.KNOWN_MODELS[args.model]
  VisionTransformer = model.partial(num_classes=dataset_info['num_classes'])
  _, params = VisionTransformer.init_by_shape(
      jax.random.PRNGKey(0),
      # Discard the "num_local_devices" dimension for initialization.
      [(batch['image'].shape[1:], batch['image'].dtype.name)])
    


  if args.mode=="from_pretrained":
    pretrained_path = os.path.join(args.vit_pretrained_dir, f'{args.model}.npz')
    print("loading pretrained from: ",pretrained_path)
    params = checkpoint.load_pretrained(
        pretrained_path=pretrained_path,
        init_params=params,
        model_config=models.CONFIGS[args.model],
        logger=logger)
  elif args.mode=="resume":
    resume_ckpt_path= join(args.resume_dir,"ckpt",args.resume_from)
    print("resuming from ",resume_ckpt_path)
    params = checkpoint.load(
        path=resume_ckpt_path
    )
  elif args.mode=="from_scratch":
    pass
  else:
    assert False, args.mode

  # pmap replicates the models over all TPUs/GPUs
  vit_fn_repl = jax.pmap(VisionTransformer.call)
  update_fn_repl = make_update_fn(VisionTransformer.call, args.accum_steps, args.weight_decay)

  # Create optimizer and replicate it over all TPUs/GPUs
  opt = momentum_clip.Optimizer(
      dtype=args.optim_dtype, grad_norm_clip=args.grad_norm_clip).create(params)
  opt_repl = flax_utils.replicate(opt)

  # Delete references to the objects that are not needed anymore
  del opt
  del params

  def copyfiles(paths):
    """Small helper to copy files to args.copy_to using tf.io.gfile."""
    if not args.copy_to:
      return
    for path in paths:
      to_path = os.path.join(args.copy_to, args.exp_name, os.path.basename(path))
      tf.io.gfile.makedirs(os.path.dirname(to_path))
      tf.io.gfile.copy(path, to_path, overwrite=True)
      logger.info(f'Copied {path} to {to_path}.')

  total_steps = args.total_steps or (
      input_pipeline.DATASET_PRESETS[args.dataset]['total_steps'])

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  lr_fn = hyper.create_learning_rate_schedule(
    total_steps, 
    args.base_lr,
    args.decay_type,
    args.warmup_steps,
    args.linear_end
  )
  resume_step=0
  if args.resume_dir:
    resume_step=int(splitext(args.resume_from)[0].split('-')[1])

  lr_iter = hyper.lr_prefetch_iter(lr_fn, resume_step, total_steps)
  update_rngs = jax.random.split(
      jax.random.PRNGKey(0), jax.local_device_count())

  # Run training loop
  writer = metric_writers.create_default_writer(tensorboard_dir, asynchronous=False)
  writer.write_hparams({k: v for k, v in vars(args).items() if v is not None})

  if args.resume_dir:
    tb_data_df=get_previous_tensorboard_data(join(args.resume_dir,"history/tb_data"))
    tb_data_df=tb_data_df[tb_data_df["Step"]<=resume_step]
    prewrite_df_to_tb(writer,tb_data_df)
    
  logger.info('Starting training loop; initial compile can take a while...')
  t0 = time.time()

  for step, batch, lr_repl in zip(
      range(resume_step + 1, total_steps),
      input_pipeline.prefetch(ds_train, args.prefetch), lr_iter):
    opt_repl, loss_repl, update_rngs = update_fn_repl(
        opt_repl, lr_repl, batch, update_rngs)
    if step == 1:
      logger.info(f'First step took {time.time() - t0:.1f} seconds.')
      t0 = time.time()
    if args.progress_every and step % args.progress_every == 0:
      writer.write_scalars(step, dict(train_loss=float(loss_repl[0]),lr=float(lr_repl[0])))
      done = step / total_steps
      logger.info(f'Step: {step}/{total_steps} {100*done:.1f}%, '
                  f'ETA: {(time.time()-t0)/done*(1-done)/3600:.2f}h')

    # Run eval step
    if ((args.eval_every and step % args.eval_every == 0) or
        (step == total_steps)):

      accuracy_test = np.mean([
          c for batch in input_pipeline.prefetch(ds_test, args.prefetch)
          for c in (
              np.argmax(vit_fn_repl(opt_repl.target, batch['image']),
                        axis=2) == np.argmax(batch['label'], axis=2)).ravel()
      ])

      lr = float(lr_repl[0])
      logger.info(f'Step: {step} '
                  f'Learning rate: {lr:.7f}, '
                  f'Test accuracy: {accuracy_test:0.5f}')
      writer.write_scalars(step, dict(accuracy_test=accuracy_test))

      ckpt_fp=os.path.join(ckpt_dir,f"step-{step}.npz")
      checkpoint.save(flax_utils.unreplicate(opt_repl.target), ckpt_fp)
      logger.info(f'Stored fine tuned checkpoint to {ckpt_fp}')


def prewrite_df_to_tb(writer,df):
  for index,row in df.iterrows():
    writer.write_scalars(row["Step"],{row["scalar_name"]:row["Value"]})
def get_previous_tensorboard_data(data_fp):
  if os.path.isfile(data_fp):
    scalar_df=pd.read_csv(data_fp,sep=",")
  elif os.path.isdir(data_fp):
    scalar_df_list=list()
    for csv_fp in Path(data_fp).glob("*.csv"):
        csv_id=csv_fp.stem
        scalar_name=csv_id.split('-')[-1]
        scalar_df=pd.read_csv(csv_fp)
        scalar_df["Step"]=scalar_df["Step"].astype(np.int64)
        scalar_df["scalar_name"]=scalar_name
        scalar_df_list.append(scalar_df)
    scalar_df=pd.concat(scalar_df_list,axis=0)
    scalar_df=scalar_df.reset_index()
  else:
      assert False
  return scalar_df
def write_args_to_yaml(args,yaml_fp):
  with open(yaml_fp,"w") as f:
    yaml.dump(args.__dict__,f)
if __name__ == '__main__':
  # Make sure tf does not allocate gpu memory.

  tf.config.experimental.set_visible_devices([], 'GPU')

  parser = flags.argparser(models.KNOWN_MODELS.keys(),
                           input_pipeline.DATASET_PRESETS.keys())

  cmd_args=parser.parse_args()

  main(cmd_args)
