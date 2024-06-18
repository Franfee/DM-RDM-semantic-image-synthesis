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
from generate_first import StackedRandomGenerator, edm_sampler, parse_int_list, preprocess_input, save_samples
from generate_second import blur_sampler
from training.image_datasets import load_data
from torch_utils import distributed as dist


#----------------------------------------------------------------------------
@click.command()

@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default="datasets/ADEChallengeData2016")
@click.option('--data_mode',     help='dataset mode', metavar='celeba|ade20k',                      type=click.Choice(['celeba', 'ade20k']), default='ade20k', show_default=True)
@click.option('--resolution',    help='image resolution  [default: varies]', metavar='INT',         type=int, default=256)
@click.option('--label_dim',     help='label_dim  [default: varies]', metavar='INT',                type=int, default=151)

@click.option('--indir',                     help='Input directory for only-second-stage sampler', metavar='DIR',     type=str, default="result/de64")
@click.option('--outdir',                    help='Where to save the output images', metavar='DIR',                   type=str, default="result")
@click.option('--seeds',                     help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--batch', 'max_batch_size',   help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=1, show_default=True)

@click.option('--sampler_stages',            help='Which stage to conduct sampler', metavar='first|second|both',      type=click.Choice(['first', 'second', 'both']), default='both')

# first stage sampler config
@click.option('--network_first',             help='Network pickle filename', metavar='PATH|URL',                      type=str,default="training_detail/ade20k/network-64-snapshot-111263.pkl")
@click.option('--num_steps_first',           help='Number of sampling steps for first stage', metavar='INT',          type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--sigma_min_first',           help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_first',           help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho_first',                 help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_first',           help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--S_churn', 'S_churn_first',  help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=40, show_default=True)
@click.option('--S_min', 'S_min_first',      help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0.05, show_default=True)
@click.option('--S_max', 'S_max_first',      help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=50, show_default=True)
@click.option('--S_noise', 'S_noise_first',  help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

# second stage sampler config
@click.option('--network_second',            help='Network pickle filename', metavar='PATH|URL',                      type=str,default="training_detail/ade20k/network-256-snapshot-010262.pkl")
@click.option('--num_steps_second',          help='Number of sampling steps for second stage', metavar='INT',         type=click.IntRange(min=1), default=200, show_default=True)
@click.option('--sigma_min_second',          help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_second',          help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--blur_sigma_max_second',     help='Maximum sigma of blurring schedule', metavar='FLOAT',              type=click.FloatRange(min=0), default=2.0, show_default=True)
@click.option('--rho_second',                help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_second',          help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--up_scale_second',           help='Scale of upsampling, default 256/64=4', metavar='FLOAT',           type=click.IntRange(min=2), default=4, show_default=True)
@click.option('--truncation_sigma_second',   help='Truncation point of noise schedule', metavar='FLOAT',              type=click.FloatRange(min=0, min_open=True), default=0.95, show_default=True)
@click.option('--truncation_t_second',       help='Truncation point of time schedule', metavar='FLOAT',               type=click.FloatRange(min=0, max=1, min_open=True), default=0.9, show_default=True)
@click.option('--s_block_second',            help='Strength of block noise addition', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.15, show_default=True)
@click.option('--s_noise_second',            help='Strength of stochasticity', metavar='FLOAT',                       type=click.FloatRange(min=0), default=0.2, show_default=True)

def main(outdir, seeds, max_batch_size, sampler_stages, 
         network_first=None, network_second=None, indir=None,
         device=torch.device('cuda'), **sampler_kwargs):
    
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
    
    if sampler_stages in ['first', 'both']:
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
    if sampler_stages in ['second', 'both']:
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
    
    if sampler_stages == 'second':
        # Preload for only-second-stage sampling.
        dist.print0(f'Preloading first stage samples from "{indir}"...')
        preload_images = []
        for batch_seeds in rank_batches:
            image_paths = [os.path.join(indir, f'{seed - seed % 1000:06d}', f'{seed:06d}.png') for seed in batch_seeds]
            batch_images = [np.array(Image.open(path)) for path in image_paths]
            batch_images = [image[np.newaxis, :, :] if image.ndim == 2 else image.transpose(2, 0, 1) for image in batch_images]
            batch_images = np.concatenate([image[np.newaxis, ...] for image in batch_images], axis=0)
            preload_images.append(batch_images)
            
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    if sampler_stages in ['first', 'both']:
        dist.print0('first stage config:', first_stage_sampler_kwargs)
    if sampler_stages in ['second', 'both']:
        dist.print0('second stage config:', second_stage_sampler_kwargs)
    
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for i, batch_seeds in tzip(range(len(rank_batches)), rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        
        ground_images, cond = next(dataset_iterator)    
        class_labels = preprocess_input(cond, device, sampler_kwargs['label_dim'])
        
        if sampler_stages in ['first', 'both']:
            # First stage generation.
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, net_first.img_channels, net_first.img_resolution, net_first.img_resolution], device=device)
            images = edm_sampler(net_first, latents, class_labels, randn_like=rnd.randn_like, **first_stage_sampler_kwargs)
        else:
            images = torch.tensor(preload_images[i], device=device, dtype=torch.float64) / 127.5 - 1
        
        # Save outputs
        save_samples(images, batch_seeds, outdir+"/de64")
        if sampler_stages == 'first':
            continue
        else:
            # Upsample for second stage generation.
            images = F.interpolate(images, 256)
            save_samples(ground_images, batch_seeds, outdir+"/gt256")
            
            
        if sampler_stages in ['second', 'both']:
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