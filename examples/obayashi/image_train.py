"""
Train a diffusion model on images.
"""
import argparse
import torch.distributed as dist
import dist_util, logger
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

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True,
                        stderrToServer=True)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    # options=args_to_dict(args, model_and_diffusion_defaults(args.super_res).keys()) # options only corresponding to model and diffusion
    # model, diffusion = create_model_and_diffusion(**options)

    options=args_to_dict(args, diffusion_defaults().keys()) # options only corresponding to model and diffusion
    diffusion = create_gaussian_diffusion(**options)

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

##### scratch #####
    if args.model_path:
        print('loading decoder')
        model_ckpt = dist_util.load_state_dict(args.model_path, map_location="cpu")

        for k  in list(model_ckpt.keys()):
            if k.startswith("transformer") and 'transformer_proj'  not in k: # total 16 transformer.resblocks
                # print(f"Removing key {k} from pretrained checkpoint") # because we are loading only decoder here, # so need to remove all encoder params
                del model_ckpt[k]
            if k.startswith("padding_embedding") or k.startswith("positional_embedding") or k.startswith("token_embedding") or k.startswith("final_ln"):
                # print(f"Removing key {k} from pretrained checkpoint")
                del model_ckpt[k]

        model.decoder.load_state_dict(
            model_ckpt   , strict=True )


    if args.encoder_path:
        print('loading encoder')
        encoder_ckpt = dist_util.load_state_dict(args.encoder_path, map_location="cpu")
        model.encoder.load_state_dict(
            encoder_ckpt   , strict=True )        

    # model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)


    all_files = _list_image_files_recursively(args.data_dir, return_full_paths=True)
    random.shuffle(all_files)
    split_indx = int(len(all_files) * args.train_ratio) + 1
    train_files = all_files[:split_indx]
    val_files = all_files[split_indx:]
    metadata_file = os.path.join(args.data_dir, 'metadata.jsonl')

    text_dict = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                text_dict.update({data['file_name'] : data['text']})

########### dataset selection
    logger.log("creating data loader...")
 
    data = load_data_sketch(
        all_files=train_files,
        text_dict=text_dict,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=True,
        low_res=args.super_res,
        uncond_p=args.uncond_p,
        mode=args.mode,
        random_crop=True,
    )

    eff_bs = max(1, args.batch_size // 2)
    eff_bs = eff_bs * 1 if args.finetune_decoder else eff_bs  # for former eff_bs=1 without this; =1 after this
    val_data = load_data_sketch(
        all_files=val_files,
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

    if args.scale_lr:
        args.lr = (
            # args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
                args.lr * args.gradient_accumulation_steps * args.batch_size * dist.get_world_size()
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    logger.log("training...")
    TrainLoop(
        model,
        options,
        diffusion,
        vae,
        tokenizer,
        text_encoder,
        data=data,
        val_data=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval=args.val_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        finetune_decoder=args.finetune_decoder,
        mode=args.mode,
        use_vgg=args.super_res,
        use_gan=args.super_res,
        uncond_p=args.uncond_p,
        super_res=args.super_res,
        opt_cls=optimizer_cls,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    ).run_loop()

 
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
        gradient_checkpointing=True,
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
