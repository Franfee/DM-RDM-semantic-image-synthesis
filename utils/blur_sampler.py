
import numpy as np
import torch

from training.blurring import dct_2d, idct_2d
from training.blurring import block_noise, get_alpha_t


def blur_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=250, sigma_min=0.008, sigma_max=80, rho=7,
    truncation_sigma=0.9, truncation_t=0.93, up_scale=4, cfg_scale=0,
    s_block=0.15, s_noise=0.2, blur_sigma_max=3,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
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

        # --------------------------------for celeba--------------------------------
        # Increase noise temporarily.
        # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        # t_hat = net.round_sigma(t_cur + gamma * t_cur)
        # x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # x_cur, t_cur = x_hat, t_hat
        # --------------------------------------------------------------------------
           
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