import copy
import functools
import os
import random

import blobfile as bf
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from glide_util import sample
import dist_util, logger
from fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from nn import update_ema
from vgg import VGG
from adv import AdversarialLoss
from resample import LossAwareSampler, UniformSampler
import glob
from script_util import write_2images
import torchvision.utils as tvu
import PIL.Image as Image
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
 
 

class TrainLoop:
    def __init__(
        self,
        model,
        glide_options,
        diffusion,
        vae,
        tokenizer,
        text_encoder,
        data,
        val_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        val_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        finetune_decoder = False,
        mode = '',
        use_vgg = False,
        use_gan = False,
        uncond_p = 0,
        super_res = 0,
        opt_cls=None,
        gradient_accumulation_steps=0,
    ):
        self.model = model.to(dist_util.dev())
        self.glide_options=glide_options
        self.diffusion = diffusion
        self.weight_dtype = torch.float16 # hardcoding as of now, to be used for vae and text encoder
        self.vae = vae.to(dist_util.dev(), dtype=self.weight_dtype) # hard coding float16 as of now as these will be inference mode
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) # factor by which vae reduces image size
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(dist_util.dev(), dtype=self.weight_dtype) # same as above
        self.data = data
        self.val_data=val_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        self.resume_checkpoint = find_resume_checkpoint(resume_checkpoint)
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if use_vgg:
            self.vgg = VGG(conv_index='22').to(dist_util.dev()) # we are not updating these parameters
            print('use perc')
        else:
            self.vgg = None

        if use_gan:
            self.adv = AdversarialLoss() # we are updating/learning the discriminator parameters here
            print('use adv')
        else:
            self.adv = None

        self.super_res = super_res
         
        self.uncond_p =uncond_p
        self.mode = mode

        self.finetune_decoder = finetune_decoder
        if finetune_decoder:
            self.optimize_model = self.model
        else:          
            self.optimize_model = self.model.encoder
         
        self.model_params = list(self.optimize_model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        # self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        self.opt = opt_cls(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        total_batch_size = self.batch_size * dist.get_world_size() * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {self.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps [Not implemented yet] = {self.gradient_accumulation_steps}")

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(th.load(resume_checkpoint, map_location="cpu"),strict=False)

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self._state_dict_to_master_params(state_dict)

        #dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location="cpu")
            try:
                self.opt.load_state_dict(state_dict)
            except:
                pass

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step   <= self.lr_anneal_steps
        ):

            batch, model_kwargs = next(self.data) # model_kwargs has ref (sketch), text (captions)

            # uncond_p = 0
            # if self.super_res:
            #     uncond_p = 0
            # elif   self.finetune_decoder:
            #     uncond_p = self.uncond_p
            # elif  self.step > self.lr_anneal_steps - 40000:
            #     uncond_p = self.uncond_p

            self.run_step(batch, model_kwargs)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            if self.step % self.val_interval == 0: # uses current ema parameters rather than saved ones on the disk
                self.val(self.step)
            self.step += 1
         
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    def run_step(self, batch, model_kwargs):
        self.forward_backward(batch, model_kwargs)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, model_kwargs):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond={n:model_kwargs[n][i:i+self.microbatch].to(dist_util.dev()) for n in model_kwargs if n  in ['ref', 'low_res']}
            # VAE (image) and CLIP (text) encoding
            # weight_dtype = th.float16 if self.use_fp16 else th.float32
            latents = self.vae.encode(micro.to(self.weight_dtype)).latent_dist.sample()  # sample # from diag_gauss distribution (because it is vae)
            micro = latents * 0.18215 # B X 4 x H/8 x W/8

            # Sketch encoding
            # enc_sketch = self.encoder(micro_cond['ref'])
            # micro = torch.cat([latents, enc_sketch], dim=1) # B X 5 X ...
            input_ids = self.tokenize_captions(model_kwargs['text'])
            padded_tokens = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
            # Get the text embedding for conditioning # emb size is 768
            micro_cond['encoder_hidden_states'] = self.text_encoder(padded_tokens.to(dist_util.dev()))[0]
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
             
            if self.step <100: # first 100 training steps, we don't enable vgg and adv as predicted image is mostly noise
                vgg_loss = None
                adv_loss = None
            else:
                vgg_loss = self.vgg
                adv_loss = self.adv
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                vgg_loss,
                adv_loss,
                model_kwargs=micro_cond,
            )
            
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
           
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def val(self, step):
        inner_model=self.ddp_model.module
        inner_model.eval()

        # disabling the below ones as more memory is needed to save copies of params for U-net
        # # copy the current model params to be loaded later
        # model_params_copy = copy.deepcopy(self.master_params)
        #
        # # load ema latest params as they are the stable params
        # self._load_params(self.ema_params[0])

        if dist.get_rank() == 0:
            print("sampling...")   

        s_path = os.path.join(logger.get_dir(), 'results')
        os.makedirs(s_path,exist_ok=True)
        guidance_scale=self.glide_options['sample_c']

        all_images = []
        all_lowres_images = []
        all_cond_images = []
        all_orig_images = []
        eff_bs = max(1, self.batch_size // 2)
        eff_bs = eff_bs * 1 if self.finetune_decoder else eff_bs # for former eff_bs=1 without this; =1 after this
        num_samples = self.glide_options['num_samples']
        image_size = self.glide_options['image_size']
        img_disp_nrow = self.glide_options['img_disp_nrow']
        while(len(all_images) * eff_bs < num_samples):

        # img_id = 0
        # while (True):
        # if img_id >= self.glide_options['num_samples']:
        #     break

            batch, model_kwargs = next(self.val_data) # uncond_p=0 here, so always real ref
            # CLIP (text) encoding
            input_ids = self.tokenize_captions(model_kwargs['text'])
            padded_tokens = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids
            # Get the text embedding for conditioning # emb size is 768
            model_kwargs['encoder_hidden_states'] = self.text_encoder(padded_tokens.to(dist_util.dev()))[0]
            if 'low_res' in model_kwargs:
                low_res = model_kwargs['low_res']
                upsampled = F.interpolate(low_res, (image_size, image_size), mode="bilinear")
                upsampled = upsampled.to(dist_util.dev())
            orig_images = batch.to(dist_util.dev())
            ref_images = model_kwargs['ref_ori'].to(dist_util.dev())
            # model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            with th.no_grad():
                samples=sample(
                    glide_model=inner_model,
                    glide_options=self.glide_options,
                    side_x=image_size // self.vae_scale_factor,
                    side_y=image_size // self.vae_scale_factor,
                    prompt=model_kwargs,
                    batch_size=eff_bs,
                    guidance_scale=guidance_scale,
                    device=dist_util.dev(),
                    prediction_respacing=self.glide_options['sample_respacing'],
                    upsample_enabled=self.glide_options['super_res'],
                    upsample_temp=0.997,
                    mode = self.mode,
                    )

            # Generated samples (output)
            samples = self.decode_latents(samples) # decode to pixel space
            gathered_samples = [th.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, samples)  # gather not supported with NCCL, but all_gather is.
            # Original images (x_0)
            gathered_orig_samples = [th.zeros_like(samples) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_orig_samples, orig_images)
            # Low resolution images
            if 'low_res' in model_kwargs:
                gathered_lowres_samples = [th.zeros_like(samples) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_lowres_samples, upsampled)
            # Sketch inputs
            elif 'ref_ori' in model_kwargs:
                gathered_cond_samples = [th.zeros_like(samples) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_cond_samples, ref_images)
            # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

            # Gather samples from all devices
            all_images.extend([samp.cpu() for samp in gathered_samples])
            all_orig_images.extend([samp.cpu() for samp in gathered_orig_samples])
            if 'low_res' in model_kwargs:
                all_lowres_images.extend([samp.cpu() for samp in gathered_lowres_samples])
            elif 'ref_ori' in model_kwargs:
                all_cond_images.extend([samp.cpu() for samp in gathered_cond_samples])

            logger.log(f"created {len(all_images) * eff_bs} samples")

        # arr = np.concatenate(all_images, axis=0)
        arr = th.vstack(all_images)
        arr = arr[: num_samples]
        arr_orig = th.vstack(all_orig_images)
        arr_orig = arr_orig[: num_samples]
        if 'low_res' in model_kwargs:
            arr_lowres = th.vstack(all_lowres_images)
            arr_ref = arr_lowres[: num_samples]
        elif 'ref_ori' in model_kwargs:
            arr_cond = th.vstack(all_cond_images)
            arr_ref = arr_cond[: num_samples]
        # Stack them row by row in order: original, ref_input, output
        arr = th.vstack([arr_orig, arr_ref, arr])

        if dist.get_rank() == 0:
            write_2images(image_outputs=arr, display_image_num=img_disp_nrow, file_name=bf.join(s_path, f"output_{(self.step + self.resume_step):06d}.jpg"))
            # if test_arr is not None:
            #     write_2images(image_outputs=test_arr, display_image_num=self.img_disp_nrow, file_name=bf.join(get_blob_logdir(), f"test_{(self.step + self.resume_step):06d}.jpg"))
        dist.barrier() # wait for rank 0 to finish writing image to filedisk
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
        inner_model.train()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.autocast('cuda', dtype=torch.float32):
            image = self.vae.decode(latents).sample
        # image = (image / 2 + 0.5).clamp(0, 1)
        # # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def tokenize_captions(self, captions_in, is_train=True):
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
        inputs = self.tokenizer(captions, max_length=self.tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        return
   

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _load_params(self, params): # load given params # be sure to pass reference to self.model.xxx
        state_dict = self._master_params_to_state_dict(params) # copy params to state_dict
        # self.model.load_state_dict(state_dict)
        self.optimize_model.load_state_dict(state_dict)

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                list(self.optimize_model.parameters()), master_params
            )
        state_dict = self.optimize_model.state_dict()
        for i, (name, _value) in enumerate(self.optimize_model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.optimize_model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    filename=filename.split('/')[-1]
    assert(filename.endswith(".pt"))
    filename=filename[:-3]
    if filename.startswith("model"):
        split = filename[5:]
    elif filename.startswith("ema"):
        split = filename.split("_")[-1]
    else:
        return 0
    try:
        return int(split)
    except ValueError:
        return 0


def get_blob_logdir():
    p=os.path.join(logger.get_dir(),"checkpoints")
    os.makedirs(p,exist_ok=True)
    return p

def find_resume_checkpoint(resume_checkpoint):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    if not resume_checkpoint:
        return None
    if "ROOT" in resume_checkpoint:
        # maybe_root=os.environ.get("AMLT_MAP_INPUT_DIR")
        # maybe_root="OUTPUT/log" if not maybe_root else maybe_root
        maybe_root=os.environ.get("ROOT")
        maybe_root=os.environ.get("LOGDIR") if not maybe_root else maybe_root
        root=os.path.join(maybe_root,"checkpoints")
        resume_checkpoint=resume_checkpoint.replace("ROOT",root)
    if "LATEST" in resume_checkpoint:
        files=glob.glob(resume_checkpoint.replace("LATEST","*.pt"))
        if not files:
            return None
        return max(files,key=parse_resume_step_from_filename)
    return resume_checkpoint



def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

