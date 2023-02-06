import os
from typing import Tuple
import dist_util
import PIL
import numpy as np
import torch as th
from script_util import (
    create_gaussian_diffusion,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# Sample from the base model.

#@th.inference_mode()
def sample(
    glide_model,
    glide_options,
    side_x,
    side_y,
    prompt,
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    upsample_enabled=False,
    upsample_temp=0.997,
    mode = '',
):

    eval_diffusion = create_gaussian_diffusion(
        diffusion_steps=glide_options["diffusion_steps"],
        learn_sigma=glide_options["learn_sigma"],
        noise_schedule=glide_options["noise_schedule"],
        predict_xstart=glide_options["predict_xstart"],
        rescale_timesteps=glide_options["rescale_timesteps"],
        rescale_learned_sigmas=glide_options["rescale_learned_sigmas"],
        timestep_respacing=prediction_respacing
    )
 
    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    cond_ref   =  prompt['ref']
    uncond_ref = th.ones_like(cond_ref) 
    
    model_kwargs = {}
    model_kwargs['ref'] =  th.cat([cond_ref, uncond_ref], 0).to(dist_util.dev())
    model_kwargs['encoder_hidden_states'] = th.cat((prompt['encoder_hidden_states'], prompt['encoder_hidden_states']), 0)

    def cfg_model_fn(x_t, ts, **kwargs): #cfg is classifier free guidance
        half = x_t[: len(x_t) // 2] # we are discarding the second half batch and working only with first half at # every reverse diffusion timestep!
        combined = th.cat([half, half], dim=0) # x_t and combined are not same because x_t two halfs are not same. Only in the begin'g they are same # we are doing this because we want cond_eps (eps(x_t|y)), uncond_eps (eps(x_t)) to have same input x_t
        model_out = eps = glide_model(combined, ts, **kwargs) # first half of kwargs is cond_ref, second half is all ones.
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
 
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps) # as per PITI paper first term should be cond_eps

        eps = th.cat([half_eps, half_eps], dim=0) # Generate two samples for next round of CFG with this same x_t.
        # return th.cat([eps, rest], dim=1)
        return eps


    if upsample_enabled: # need to work on this!
        model_kwargs['low_res'] = prompt['low_res'].to(dist_util.dev())
        noise = th.randn((batch_size, 3, side_y, side_x), device=device) * upsample_temp
        model_fn = glide_model # just use the base model, no need for CFG.
        model_kwargs['ref'] =  model_kwargs['ref'][:batch_size] # so we are sending both low_res and ref during sampling

        samples = eval_diffusion.p_sample_loop(
        model_fn,
        (batch_size, 3, side_y, side_x),  # only thing that's changed
        noise=noise,
        device=device,
        clip_denoised=True,
        progress=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]

    else:
        model_fn = cfg_model_fn # so we use CFG for the base model.
        # noise = th.randn((batch_size, 3, side_y, side_x), device=device)
        noise = th.randn((batch_size, 4, side_y, side_x), device=device) # VAE has 4 output channels
        noise = th.cat([noise, noise], 0)
        samples = eval_diffusion.p_sample_loop(
            model_fn,
            # (full_batch_size, 3, side_y, side_x),  # only thing that's changed
            (full_batch_size, 4, side_y, side_x),  # VAE has 4 output channels
            noise=noise,
            device=device,
            clip_denoised=True,
            progress=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
    
    return samples

 