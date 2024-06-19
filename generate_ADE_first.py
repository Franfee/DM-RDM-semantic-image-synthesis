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

import dnnlib
from training.image_datasets import load_data
from torch_utils import distributed as dist


def preprocess_input(data, device, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = torch.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    return input_semantics.to(device)


def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=256, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=40, S_min=0.05, S_max=50, S_noise=1.003, cfg_scale=0
):
    
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
        # Euler step.
        if cfg_scale > 1:
            denoised_cond = net(x_hat, t_hat, class_labels).to(torch.float64)
            denoised_uncond = net(x_hat, t_hat, torch.zeros_like(class_labels)).to(torch.float64)
            denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
        else:
            denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        
        # ## adding SDE
        # g_t = 0.05 * t_next
        # a_t = ((t_next**2 - g_t**2) / t_hat**2) ** 0.5

        # x_next = x_hat + (a_t - 1) * t_hat * d_cur + g_t * randn_like(x_next)

        # # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     denoised = net(x_next, t_next, class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next
        #     x_next = x_hat + (a_t - 1) * t_hat * (0.5 * d_cur + 0.5 * d_prime) + g_t * randn_like(x_next)
        
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # Apply 2nd order correction.
        if i < num_steps - 1:
            if cfg_scale > 1:
                denoised_cond = net(x_next, t_next, class_labels).to(torch.float64)
                denoised_uncond = net(x_next, t_next, torch.zeros_like(class_labels)).to(torch.float64)
                denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
            else:
                denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


#----------------------------------------------------------------------------
# Sample saver
def save_samples(images, batch_seeds, out_dir):
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    for seed, image_np in zip(batch_seeds, images_np):
        image_dir = os.path.join(out_dir, f'{seed - seed % 1000:06d}')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f'{seed:06d}.png')
        if image_np.shape[2] == 1:
            Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
        else:
            Image.fromarray(image_np, 'RGB').save(image_path)


#----------------------------------------------------------------------------
@click.command()
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, default="datasets/ADEChallengeData2016")
@click.option('--data_mode',     help='dataset mode', metavar='celeba|ade20k',                      type=click.Choice(['celeba', 'ade20k']), default='ade20k', show_default=True)
@click.option('--resolution',    help='image resolution  [default: varies]', metavar='INT',         type=int, default=64)
@click.option('--label_dim',     help='label_dim  [default: varies]', metavar='INT',                type=int, default=151)

@click.option('--outdir',                    help='Where to save the output images', metavar='DIR',                   type=str, default="result")
@click.option('--seeds',                     help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--batch', 'max_batch_size',   help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=1, show_default=True)

# first stage sampler config
@click.option('--network_first',             help='Network pickle filename', metavar='PATH|URL',                      type=str,default="training_detail/network-64-snapshot-111263.pkl")
@click.option('--num_steps_first',           help='Number of sampling steps for first stage', metavar='INT',          type=click.IntRange(min=1), default=120, show_default=True)
@click.option('--sigma_min_first',           help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max_first',           help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho_first',                 help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--cfg_scale_first',           help='Scale of classifier-free guidance', metavar='FLOAT',               type=click.FloatRange(min=0), default=3.5, show_default=True)
@click.option('--S_churn', 'S_churn_first',  help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min_first',      help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max_first',      help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise_first',  help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)


def main(outdir, seeds, max_batch_size, network_first=None,
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
            
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    dist.print0('first stage config:', first_stage_sampler_kwargs)
   
    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for i, batch_seeds in tzip(range(len(rank_batches)), rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        
        ground_images, cond = next(dataset_iterator)
        save_samples(ground_images, batch_seeds, outdir+"/gt64")

        class_labels = preprocess_input(cond, device, sampler_kwargs['label_dim'])
        
        # First stage generation.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net_first.img_channels, net_first.img_resolution, net_first.img_resolution], device=device)
        images = edm_sampler(net_first, latents, class_labels, randn_like=rnd.randn_like, **first_stage_sampler_kwargs)

        # Save outputs
        save_samples(images, batch_seeds, outdir+"/de64") 
        
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')    

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------