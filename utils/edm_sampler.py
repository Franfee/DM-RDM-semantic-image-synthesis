

import torch
import numpy as np


@torch.no_grad()
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=256, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=40, S_min=0.05, S_max=50, S_noise=1.003, cfg_scale=0, sde_scale=0.05
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
        
        # --------------------------------way 1-------------------------------------
        # ## adding SDE for celeba
        # g_t = sde_scale * t_next
        # a_t = ((t_next**2 - g_t**2) / t_hat**2) ** 0.5

        # x_next = x_hat + (a_t - 1) * t_hat * d_cur + g_t * randn_like(x_next)

        # # Apply 2nd order correction.
        # if i < num_steps - 1:
        #     if cfg_scale > 1:
        #         denoised_cond = net(x_next, t_next, class_labels).to(torch.float64)
        #         denoised_uncond = net(x_next, t_next, torch.zeros_like(class_labels)).to(torch.float64)
        #         denoised = cfg_scale * denoised_cond - (cfg_scale - 1) * denoised_uncond
        #     else:
        #         denoised = net(x_next, t_next, class_labels).to(torch.float64)
        #     d_prime = (x_next - denoised) / t_next
        #     x_next = x_hat + (a_t - 1) * t_hat * (0.5 * d_cur + 0.5 * d_prime) + g_t * randn_like(x_next)
        # --------------------------------------------------------------------------


        # --------------------------------way 2-------------------------------------
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
        # --------------------------------------------------------------------------
    
    return x_next


