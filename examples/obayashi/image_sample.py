"""
Train a diffusion model on images.
"""
import argparse
import torch.distributed as dist
import dist_util, logger
from glide_util import sample
# from image_datasets_mask import load_data_mask
from image_datasets_sketch import load_data_sketch
# from image_datasets_depth import load_data_depth
from resample import create_named_schedule_sampler
from script_util import (
    # model_and_diffusion_defaults,
    diffusion_defaults,
    model_defaults,
    # create_model_and_diffusion,
    create_gaussian_diffusion,
    create_model,
    args_to_dict,
    add_dict_to_argparser,)
from train_util import TrainLoop
import torch
import json
import sys, os
import random
from create_metadata import _list_image_files_recursively
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
# sys.path.append("~/pycharm-professional/debug-eggs/pydevd-pycharm.egg")
import torch.nn.functional as F
from script_util import write_2images
from script_util import write_2images_onebyone
import numpy as np
from train_util import find_resume_checkpoint

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True,
                        stderrToServer=True)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    # options=args_to_dict(args, model_and_diffusion_defaults(args.super_res).keys()) # options only corresponding to model and diffusion
    # model, diffusion = create_model_and_diffusion(**options)

    # options=args_to_dict(args, diffusion_defaults().keys()) # options only corresponding to model and diffusion
    # diffusion = create_gaussian_diffusion(**options)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained( # emb_size=768
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    vae.to(dist_util.dev())
    text_encoder.to(dist_util.dev())

    options=args_to_dict(args, model_defaults().keys()) # options only corresponding to model and diffusion
    model = create_model(decoder=unet, **options)

    options=args_to_dict(args) # ALL the options
    if dist.get_rank() == 0:
        logger.save_args(options)

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    model_path = args.model_path if args.model_path else find_resume_checkpoint(args.resume_checkpoint)
##### scratch #####
    if model_path:
        print('loading model')
        # two methods to sync params across ranks after loading data to rank 0: (i) use mpi.COMM_WORLD.bcast as in dist_util.load_state_dict() or use dist_util.sync_params() (uses dist.bcast instead of mpi.COMM_WORLD.bcast, so is better at handling large valued params)
        # Method 1 (mpi giving overflowerror)
        # model_ckpt = dist_util.load_state_dict(model_path, map_location="cpu")
        # model.load_state_dict( model_ckpt, strict=True )
        # Method 2 (uses dist.bcast instead of mpi)
        if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location="cpu"),strict=True)

        model.to(dist_util.dev())
        model.eval()
        dist_util.sync_params(model.parameters())


    if args.encoder_path:
        print('loading encoder')
        encoder_ckpt = dist_util.load_state_dict(args.encoder_path, map_location="cpu")
        model.encoder.load_state_dict(
            encoder_ckpt   , strict=True )        

    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    all_files = _list_image_files_recursively(args.data_dir, return_full_paths=True)
    metadata_file = os.path.join(args.data_dir, 'metadata.jsonl')

    text_dict = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                text_dict.update({data['file_name'] : data['text']})

########### dataset selection
    logger.log("creating data loader...")
 
    eff_bs = max(1, args.batch_size // 2)
    # eff_bs = eff_bs * 1 if args.finetune_decoder else eff_bs  # for former eff_bs=1 without this; =1 after this
    val_data = load_data_sketch(
        all_files=all_files,
        text_dict=text_dict,
        batch_size=eff_bs,
        image_size=args.image_size,
        train=False,
        deterministic=True,
        low_res=args.super_res,
        uncond_p=0.,
        mode=args.mode,
        random_crop=False,
    )

    if dist.get_rank() == 0:
        print("sampling...")

    # s_path = os.path.join(logger.get_dir(), 'results_infer')
    s_path = logger.get_dir()
    os.makedirs(s_path, exist_ok=True)
    guidance_scale = args.sample_c
    # options=args_to_dict(args) # ALL the options


    all_images = []
    all_lowres_images = []
    all_cond_images = []
    all_orig_images = []
    all_filename_prompt_dict = []
    # eff_bs = max(1, args.batch_size // 2)
    num_samples = args.num_samples
    image_size = args.image_size
    img_disp_nrow = args.img_disp_nrow
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)  # factor by which vae reduces image size
    while (len(all_images) * eff_bs < num_samples):

        # img_id = 0
        # while (True):
        # if img_id >= self.glide_options['num_samples']:
        #     break

        batch, model_kwargs = next(val_data)  # uncond_p=0 here, so always real ref
        # CLIP (text) encoding
        input_ids = tokenize_captions(tokenizer, model_kwargs['text'])
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
        # Get the text embedding for conditioning # emb size is 768
        model_kwargs['encoder_hidden_states'] = text_encoder(padded_tokens.to(dist_util.dev()))[0]
        if 'low_res' in model_kwargs:
            low_res = model_kwargs['low_res']
            upsampled = F.interpolate(low_res, (image_size, image_size), mode="bilinear")
            upsampled = upsampled.to(dist_util.dev())
        orig_images = batch.to(dist_util.dev())
        ref_images = model_kwargs['ref_ori'].to(dist_util.dev())
        # model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        filename_prompt_dict = [{'file_name': os.path.basename(x), 'text': y} for x,y in zip(model_kwargs['path'], model_kwargs['text'])]
        with torch.no_grad():
            samples = sample(
                glide_model=model,
                glide_options=options,
                side_x=image_size // vae_scale_factor,
                side_y=image_size // vae_scale_factor,
                prompt=model_kwargs,
                batch_size=eff_bs,
                guidance_scale=guidance_scale,
                device=dist_util.dev(),
                prediction_respacing=args.sample_respacing,
                upsample_enabled=args.super_res,
                upsample_temp=0.997,
                mode=args.mode,
            )

        # Generated samples (output)
        samples = decode_latents(vae, samples)  # decode to pixel space
        gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL, but all_gather is.
        # Original images (x_0)
        gathered_orig_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_orig_samples, orig_images)
        # Low resolution images
        if 'low_res' in model_kwargs:
            gathered_lowres_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_lowres_samples, upsampled)
        # Sketch inputs
        elif 'ref_ori' in model_kwargs:
            gathered_cond_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_cond_samples, ref_images)
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_dicts = [[] for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_dicts, filename_prompt_dict)

        # Gather samples from all devices
        all_images.extend([samp.cpu() for samp in gathered_samples])
        all_orig_images.extend([samp.cpu() for samp in gathered_orig_samples])
        if 'low_res' in model_kwargs:
            all_lowres_images.extend([samp.cpu() for samp in gathered_lowres_samples])
        elif 'ref_ori' in model_kwargs:
            all_cond_images.extend([samp.cpu() for samp in gathered_cond_samples])
        for l in gathered_dicts:
            all_filename_prompt_dict.extend(l)

        logger.log(f"created {len(all_images) * eff_bs} samples")

    # arr = np.concatenate(all_images, axis=0)
    arr = torch.vstack(all_images)
    arr = arr[: num_samples]
    arr_orig = torch.vstack(all_orig_images)
    arr_orig = arr_orig[: num_samples]
    all_filename_prompt_dict = all_filename_prompt_dict[: num_samples]
    if 'low_res' in model_kwargs:
        arr_lowres = torch.vstack(all_lowres_images)
        arr_ref = arr_lowres[: num_samples]
    elif 'ref_ori' in model_kwargs:
        arr_cond = torch.vstack(all_cond_images)
        arr_ref = arr_cond[: num_samples]

    if dist.get_rank() == 0:
        # write prompt file to destination
        with open(os.path.join(s_path, 'prompts.txt'), 'w') as f:
            for dt in all_filename_prompt_dict:
                file_name = dt['file_name']
                text = dt['text']
                f.write(f'{file_name}: {text}\n')
            f.close()
        # write images to destination
        if img_disp_nrow == 1: # save each image separately to know the relation between image and prompt
            write_2images_onebyone(arr_orig, arr_ref, arr, all_filename_prompt_dict, s_path)
        else:
            # Stack them row by row in order: original, ref_input, output
            arr = torch.vstack([arr_orig, arr_ref, arr])
            write_2images(image_outputs=arr, display_image_num=img_disp_nrow,
                          file_name=os.path.join(s_path, f"output.jpg"))
        # if test_arr is not None:
        #     write_2images(image_outputs=test_arr, display_image_num=self.img_disp_nrow, file_name=bf.join(get_blob_logdir(), f"test_{(self.step + self.resume_step):06d}.jpg"))
    dist.barrier()  # wait for rank 0 to finish writing image to filedisk
    # # samples = samples.cpu()
    # ref = model_kwargs['ref_ori']
    # # LR = model_kwargs['low_res'].cpu()
    #
    # for i in range(samples.size(0)):
    #     out_path = os.path.join(s_path, f"{dist.get_rank()}_{img_id}_step{step}_{guidance_scale}_output.png")
    #     tvu.save_image(
    #         (samples[i]+1)*0.5, out_path)
    #
    #     out_path = os.path.join(s_path, f"{dist.get_rank()}_{img_id}_step{step}_{guidance_scale}_gt.png")
    #     tvu.save_image(
    #         (batch[i]+1)*0.5, out_path)
    #
    #     out_path = os.path.join(s_path, f"{dist.get_rank()}_{img_id}_step{step}_{guidance_scale}_ref.png")
    #     tvu.save_image(
    #         (ref[i]+1)*0.5, out_path)
    #
    #     # out_path = os.path.join(s_path, f"{dist.get_rank()}_{img_id}_step{step}_{guidance_scale}_lr.png")
    #     # tvu.save_image(
    #     #     (LR[i]+1)*0.5, out_path)
    #
    #     img_id += 1

    # # load latest model params
    # self._load_params(model_params_copy)
    # inner_model.train()


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
def decode_latents(vae, latents):
    latents = 1 / 0.18215 * latents
    with torch.autocast('cuda', dtype=torch.float32):
        image = vae.decode(latents).sample
    # image = (image / 2 + 0.5).clamp(0, 1)
    # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image


def tokenize_captions(tokenizer, captions_in, is_train=True):
    captions = []
    for caption in captions_in:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"{captions} should contain either strings or lists of strings."
            )
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
    input_ids = inputs.input_ids
    return input_ids


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        model_path="",
        encoder_path="",
        schedule_sampler="uniform",
        image_size=256,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=200,
        save_interval=5000,
        val_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        super_res=0,
        sample_c=1.,
        sample_respacing="100",
        uncond_p=0.2,
        num_samples=1,
        finetune_decoder=False,
        mode="",
        pretrained_model_name_or_path='stabilityai/stable-diffusion-2-depth',
        revision=None, # revision of hugging face pretrained models
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        scale_lr=False,
        train_ratio=0.8,
        cache_dir=None,
        seed=0,
        use_8bit_adam=False,
        img_disp_nrow=16,
        )
    # defaults.update(model_and_diffusion_defaults())
    defaults.update(model_defaults())
    defaults.update(diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
