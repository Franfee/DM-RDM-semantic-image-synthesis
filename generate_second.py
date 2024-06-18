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

from generate_first import StackedRandomGenerator, parse_int_list, preprocess_input, save_samples
from training.image_datasets import load_data
from torch_utils import distributed as dist
from training.blurring import dct_2d, idct_2d
from training.blurring import block_noise, get_alpha_t


def blur_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=250, sigma_min=0.008, sigma_max=80, rho=7,
    truncation_sigma=0.9, truncation_t=0.93, up_scale=4, cfg_scale=0,
    s_block=0.15, s_noise=0.2, blur_sigma_max=3
):
    """
    truncation_sigma: Truncation point of noise schedule
    up_scale: Scale of upsampling, default 256/64=4
    cfg_scale: Scale of classifier-free guidance
    s_block: Scale of block noise addition
    s_noise: Scale of stochasticity in sampler
    """
    
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Time step discretization.S
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    idx_after_truncation = 0
    while t_steps[idx_after_truncation] >= truncation_sigma:
        idx_after_truncation += 1
    t_steps = t_steps[idx_after_truncation:]
    num_steps = len(t_steps)
    
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64)
    x_cur = None
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        
        if x_cur is None:
            if s_block > 0:
                x_cur = x_next + randn_like(x_next) * t_cur + s_block * block_noise(latents, randn_like, up_scale, device=latents.device) * t_cur
            else:
                x_cur = x_next + randn_like(x_next) * t_cur
        else:
            x_cur = x_next
            
        # Euler step.
        if cfg_scale > 1:
            denoised_cond = net(x_cur, t_cur, class_labels).to(torch.float64)
            denoised_uncond = net(x_cur, t_cur, torch.zeros_like(class_labels)).to(torch.float64)
            denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
        else:
            denoised = net(x_cur, t_cur, class_labels).to(torch.float64)
            
        if i == num_steps - 1:
            return denoised
        
        alpha_next = get_alpha_t(t_next, up_scale, latents.device, prob_length=truncation_t, blur_sigma_max=blur_sigma_max)
        alpha_cur = get_alpha_t(t_cur, up_scale, latents.device, prob_length=truncation_t, blur_sigma_max=blur_sigma_max)
        
        u_cur = dct_2d(x_cur, up_scale, norm='ortho')
        u_0 = dct_2d(denoised, up_scale, norm='ortho')
        d_cur = (u_cur - u_0) / t_cur
        
        gamma = (1 - s_noise**2)**0.5 * t_next / t_cur
        
        if s_block > 0:
            x_next = idct_2d((alpha_next + gamma - gamma * alpha_cur) * u_cur \
                + t_cur * (gamma * alpha_cur - alpha_next) * d_cur, up_scale, norm='ortho') \
                + s_noise * t_next * (randn_like(x_next) + s_block * block_noise(latents, randn_like, up_scale, device=latents.device))
        else:
            x_next = idct_2d((alpha_next + gamma - gamma * alpha_cur) * u_cur \
                + t_cur * (gamma * alpha_cur  - alpha_next) * d_cur, up_scale, norm='ortho') \
                + s_noise * t_next * randn_like(x_next)
                
        # Apply 2nd order correction.
        if i < num_steps - 1:
            if cfg_scale > 1:
                denoised_cond = net(x_next, t_next, class_labels).to(torch.float64)
                denoised_uncond = net(x_next, t_next, torch.zeros_like(class_labels)).to(torch.float64)
                denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
            else:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
                  
            u_next = dct_2d(x_next, up_scale, norm='ortho')
            u_0 = dct_2d(denoised, up_scale, norm='ortho')
            d_prime = (u_next - u_0) / t_next
            if s_block > 0:
                x_next = idct_2d((alpha_next + gamma - gamma * alpha_cur) * u_cur \
                    + t_cur * (gamma * alpha_cur - alpha_next) * (d_cur + d_prime) / 2, up_scale, norm='ortho') \
                    + s_noise * t_next * (randn_like(x_next) + s_block * block_noise(latents, randn_like, up_scale, device=latents.device))
            else:
                x_next = idct_2d((alpha_next + gamma - gamma * alpha_cur) * u_cur \
                    + t_cur * (gamma * alpha_cur - alpha_next) * (d_cur + d_prime) / 2, up_scale, norm='ortho') \
                    + s_noise * t_next * randn_like(x_next)
                    
    return x_next

#----------------------------------------------------------------------------

@click.command()

@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default="/root/autodl-tmp/CelebA-HQ")
@click.option('--data_mode',     help='dataset mode', metavar='celeba|ade20k',                      type=click.Choice(['celeba', 'ade20k']), default='celeba', show_default=True)
@click.option('--resolution',    help='image resolution  [default: varies]', metavar='INT',         type=int, default=256)
@click.option('--label_dim',     help='label_dim  [default: varies]', metavar='INT',                type=int, default=19)

@click.option('--indir',                     help='Input directory for only-second-stage sampler', metavar='DIR',     type=str, default="result/de64")
@click.option('--outdir',                    help='Where to save the output images', metavar='DIR',                   type=str, default="result")
@click.option('--seeds',                     help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--batch', 'max_batch_size',   help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=1, show_default=True)

# second stage sampler config
@click.option('--network_second',            help='Network pickle filename', metavar='PATH|URL',                      type=str,default="/root/autodl-tmp/training_detail/network-256-snapshot-012101.pkl")
@click.option('--num_steps_second',          help='Number of sampling steps for second stage', metavar='INT',         type=click.IntRange(min=1), default=260, show_default=True)
@click.option('--sigma_min_second',          help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_second',          help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--blur_sigma_max_second',     help='Maximum sigma of blurring schedule', metavar='FLOAT',              type=click.FloatRange(min=0), default=3.0, show_default=True)
@click.option('--rho_second',                help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_second',          help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=3.5, show_default=True)
@click.option('--up_scale_second',           help='Scale of upsampling, default 256/64=4', metavar='FLOAT',           type=click.IntRange(min=2), default=4, show_default=True)
@click.option('--truncation_sigma_second',   help='Truncation point of noise schedule', metavar='FLOAT',              type=click.FloatRange(min=0, min_open=True), default=0.95, show_default=True)
@click.option('--truncation_t_second',       help='Truncation point of time schedule', metavar='FLOAT',               type=click.FloatRange(min=0, max=1, min_open=True), default=0.9, show_default=True)
@click.option('--s_block_second',            help='Strength of block noise addition', metavar='FLOAT',                type=click.FloatRange(min=0), default=0.15, show_default=True)
@click.option('--s_noise_second',            help='Strength of stochasticity', metavar='FLOAT',                       type=click.FloatRange(min=0), default=0.23, show_default=True)

def main(outdir, seeds, max_batch_size, sampler_stages, network_second=None, indir=None,
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
    )

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
   
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
    dist.print0('second stage config:', second_stage_sampler_kwargs)

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
    
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for i, batch_seeds in tzip(range(len(rank_batches)), rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        
        ground_images, cond = next(dataset_iterator)
        save_samples(ground_images, batch_seeds, outdir+"/gt256")    
        class_labels = preprocess_input(cond, device, sampler_kwargs['label_dim'])
        
        # First stage generation.
        rnd = StackedRandomGenerator(device, batch_seeds)
        images = torch.tensor(preload_images[i], device=device, dtype=torch.float64) / 127.5 - 1

        # Upsample for second stage generation.
        images = F.interpolate(images, 256)
    
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