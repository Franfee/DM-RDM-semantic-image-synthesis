# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import pickle
from PIL import Image

from tqdm.contrib import tzip

import torch
import numpy as np
import torch.nn.functional as F

import dnnlib

from training.image_datasets import load_data
from torch_utils import distributed as dist
from utils.blur_sampler import blur_sampler
from utils.edm_sampler import edm_sampler
from utils.preprocess import preprocess_input
from utils.utils import StackedRandomGenerator, parse_int_list, save_samples


#----------------------------------------------------------------------------
@click.command()

@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default="/root/autodl-tmp/CelebA-HQ")
@click.option('--data_mode',     help='dataset mode', metavar='celeba|ade20k',                      type=click.Choice(['celeba', 'ade20k']), default='celeba', show_default=True)
@click.option('--resolution',    help='image resolution  [default: varies]', metavar='INT',         type=int, default=256)
@click.option('--label_dim',     help='label_dim  [default: varies]', metavar='INT',                type=int, default=19)

@click.option('--outdir',                    help='Where to save the output images', metavar='DIR',                   type=str, default="result")
@click.option('--seeds',                     help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--batch', 'max_batch_size',   help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=1, show_default=True)

# first stage sampler config
@click.option('--network_first',             help='Network pickle filename', metavar='PATH|URL',                      type=str,default="training_detail/celeba/network-64-snapshot-111263.pkl")
@click.option('--num_steps_first',           help='Number of sampling steps for first stage', metavar='INT',          type=click.IntRange(min=1), default=120, show_default=True)
@click.option('--sigma_min_first',           help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_first',           help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho_first',                 help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_first',           help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=1, show_default=True)

@click.option('--sde_scale_first',           help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=0.05, show_default=True)

# second stage sampler config
@click.option('--network_second',            help='Network pickle filename', metavar='PATH|URL',                      type=str,default="training_detail/celeba/network-256-snapshot-010262.pkl")
@click.option('--num_steps_second',          help='Number of sampling steps for second stage', metavar='INT',         type=click.IntRange(min=1), default=150, show_default=True)
@click.option('--sigma_min_second',          help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_second',          help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--blur_sigma_max_second',     help='Maximum sigma of blurring schedule', metavar='FLOAT',              type=click.FloatRange(min=0), default=2.0, show_default=True)
@click.option('--rho_second',                help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_second',          help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--up_scale_second',           help='Scale of upsampling, default 256/64=4', metavar='FLOAT',           type=click.IntRange(min=2), default=4, show_default=True)
@click.option('--truncation_sigma_second',   help='Truncation point of noise schedule', metavar='FLOAT',              type=click.FloatRange(min=0, min_open=True), default=0.95, show_default=True)
@click.option('--truncation_t_second',       help='Truncation point of time schedule', metavar='FLOAT',               type=click.FloatRange(min=0, max=1, min_open=True), default=1.05, show_default=True)
@click.option('--s_block_second',            help='Strength of block noise addition', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.15, show_default=True)
@click.option('--s_noise_second',            help='Strength of stochasticity', metavar='FLOAT',                       type=click.FloatRange(min=0), default=0.2, show_default=True)

def main(outdir, seeds, max_batch_size, network_first=None, network_second=None, device=torch.device('cuda'), **sampler_kwargs):
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    # Load dataset.
    dist.print0('Loading Val dataset...')
    dataset_iterator = load_data(
        dataset_mode=sampler_kwargs['data_mode'],
        data_dir=sampler_kwargs['data'],
        batch_size=max_batch_size,
        image_size=sampler_kwargs['resolution'],
        class_cond=True,
        is_train=False,
        deterministic=True,
        random_flip=False
    )

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    
    dist.print0(f'Loading first stage network from "{network_first}"...')
    
    assert network_first.endswith('pkl') or network_first.endswith('pt'), "Unknown format of the ckpt filename"
    if network_first.endswith('.pkl'):
        with dnnlib.util.open_url(network_first, verbose=(dist.get_rank() == 0)) as f:
            net_first = pickle.load(f)['ema'].to(device)
    elif network_first.endswith('.pt'):
        data = torch.load(network_first, map_location=torch.device('cpu'))
        net_first = data['ema'].eval().to(device)
    
    first_stage_sampler_kwargs = {
        k[:-6]: v for k, v in sampler_kwargs.items() if k.endswith('_first') and v is not None
    }

    dist.print0(f'Loading second stage network from "{network_second}"...')
    
    assert network_second.endswith('pkl') or network_second.endswith('pt'), "Unknown format of the ckpt filename"
    if network_second.endswith('.pkl'):
        with dnnlib.util.open_url(network_second, verbose=(dist.get_rank() == 0)) as f:
            net_second = pickle.load(f)['ema'].to(device)
    elif network_second.endswith('.pt'):
        data = torch.load(network_second, map_location=torch.device('cpu'))
        net_second = data['ema'].eval().to(device)
    
    second_stage_sampler_kwargs = {
        k[:-7]: v for k, v in sampler_kwargs.items() if k.endswith('_second') and v is not None
    }
            
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    dist.print0('first stage config:', first_stage_sampler_kwargs)
    dist.print0('second stage config:', second_stage_sampler_kwargs)
    
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for i, batch_seeds in tzip(range(len(rank_batches)), rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        
        # preprocess_input
        ground_images, cond = next(dataset_iterator)    
        class_labels = preprocess_input(cond, device, sampler_kwargs['label_dim'])
        
        # First stage generation.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net_first.img_channels, net_first.img_resolution, net_first.img_resolution], device=device)
        images = edm_sampler(net_first, latents, class_labels, randn_like=rnd.randn_like, **first_stage_sampler_kwargs)
        
        # Save outputs
        save_samples(images, batch_seeds, outdir+"/de64")

        # Upsample for second stage generation.
        images = F.interpolate(images, 256)
        save_samples(ground_images, batch_seeds, outdir+"/gt256")

        # Second stage generation.
        images = blur_sampler(net_second, images, class_labels, randn_like=rnd.randn_like, **second_stage_sampler_kwargs)
        
        save_samples(images, batch_seeds, outdir+"/de256")
        
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------