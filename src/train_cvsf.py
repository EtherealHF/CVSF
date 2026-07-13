# ============================================
# File: src/train_cvsf.py
# ============================================
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import warnings
warnings.filterwarnings("ignore")

import logging
import gc
import re
import random

import lpips
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from de_net import DEResNet
from cvsf import CVSF
from my_utils.training_utils import parse_args_paired_training, ExposurePairedDataset
from my_utils.stage1_img_encoder import load_stage1_img_enc, Stage1ImageEncoderWrapper
from dwt_loss import DWTLoss


def build_img_enc_fn(args):
    """
    Build the Stage-1 image encoder used by CVSF.
    The returned module should output vectors with dimension args.stage1_embed_dim.
    """
    try:
        from my_utils.image_encoder import ImageEncoder
        return ImageEncoder(
            backbone=args.stage1_backbone,
            out_dim=args.stage1_embed_dim,
            pretrained=False,
        )
    except Exception as e:
        raise ImportError(
            "Failed to build Stage1 ImageEncoder. "
            "Please modify build_img_enc_fn() to import your actual ImageEncoder class."
        ) from e


def build_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "training.log"),
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )
    return logging.getLogger(__name__)


def collect_trainable_params(net_sr, accelerator):
    """Collect Stage-2 trainable parameters while keeping Stage-1 frozen."""
    layers_to_opt = []

    layers_to_opt += list(net_sr.vae_block_embeddings.parameters())
    layers_to_opt += list(net_sr.vae_de_mlp_1.parameters())
    layers_to_opt += list(net_sr.vae_de_mlp_2.parameters())
    layers_to_opt += list(net_sr.vae_block_mlp.parameters())
    layers_to_opt += list(net_sr.vae_fuse_mlp_0.parameters())
    layers_to_opt += list(net_sr.vae_fuse_mlp_1.parameters())
    layers_to_opt += list(net_sr.vae_fuse_mlp_2.parameters())
    layers_to_opt += list(net_sr.unet_processor.parameters())
    layers_to_opt += list(net_sr.eeg_mlp.parameters())
    layers_to_opt += list(net_sr.prior_mapper.parameters())

    if hasattr(net_sr, "wpenet"):
        layers_to_opt += list(net_sr.wpenet.parameters())

    for n, p in net_sr.unet.named_parameters():
        if ("adapter_blocks" in n) or ("lora" in n):
            layers_to_opt.append(p)
    layers_to_opt += list(net_sr.unet.conv_in.parameters())

    for n, p in net_sr.vae.named_parameters():
        if "lora" in n:
            layers_to_opt.append(p)

    # 鍘婚噸
    uniq = []
    seen = set()
    for p in layers_to_opt:
        if id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))

    n_params = sum(p.numel() for p in uniq if p.requires_grad)
    accelerator.print(f"[TrainMode] opt_param_count={n_params}")

    if hasattr(net_sr, "stage1_img_enc") and net_sr.stage1_img_enc is not None:
        n_stage1 = sum(p.numel() for p in net_sr.stage1_img_enc.parameters() if p.requires_grad)
        accelerator.print(f"[Stage1 img_enc] trainable_param_count={n_stage1} (expected 0)")

    return uniq


def main(args):
    # ================== 1. 閰嶇疆 & Accelerator ==================
    config = OmegaConf.load(args.base_config)

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # ================== 2. 杈撳嚭鐩綍 & 鏃ュ織 ==================
    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "visual"), exist_ok=True)

    logger = build_logger(args.output_dir) if accelerator.is_main_process else logging.getLogger(__name__)

    # ================== 3. 鍔犺浇 Stage1 image encoder ==================
    if not hasattr(args, "stage1_img_encoder_ckpt") or args.stage1_img_encoder_ckpt is None:
        raise ValueError("Please provide --stage1_img_encoder_ckpt")

    accelerator.print(f"[Stage1] loading image encoder from: {args.stage1_img_encoder_ckpt}")

    raw_img_enc = load_stage1_img_enc(
        build_img_enc_fn=lambda: build_img_enc_fn(args),
        ckpt_path=args.stage1_img_encoder_ckpt,
        device="cpu",
        strict=True,
    )

    stage1_img_enc = Stage1ImageEncoderWrapper(
        img_enc=raw_img_enc,
        image_size=args.stage1_input_size,
        freeze=True,
    )

    # ================== 4. 缃戠粶瀹氫箟 ==================
    net_de = DEResNet(num_in_ch=3, num_degradation=2)
    net_de.load_model(args.de_net_path)
    net_de = net_de.cuda()
    net_de.eval()

    net_sr = CVSF(
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_path=sd_path,
        pretrained_path=args.pretrained_path,
        ablation_mode=args.ablation_mode,
        lora_zero_parts=getattr(args, "lora_zero_parts", False),

        stage1_img_enc=stage1_img_enc,
        freeze_stage1_img_enc=True,
        aligned_dim=args.stage1_embed_dim,

        prior_map_size=args.prior_map_size,
        prior_channels=args.prior_channels,
    )
    net_sr.set_train()

    accelerator.print(
        f"[Config] mode={args.ablation_mode}, "
        f"stage1_backbone={args.stage1_backbone}, "
        f"stage1_embed_dim={args.stage1_embed_dim}, "
        f"stage1_input_size={args.stage1_input_size}, "
        f"prior_map_size={args.prior_map_size}, "
        f"prior_channels={args.prior_channels}, "
        f"lora_zero_parts={getattr(args, 'lora_zero_parts', False)}"
    )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            # net_sr.unet.enable_xformers_memory_efficient_attention()
            accelerator.print("xformers requested, but skipped in this script")
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_sr.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 4.3 鍒ゅ埆鍣?    if args.gan_disc_type == "vagan":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(
            cv_type="dino",
            output_type="conv_multi_level",
            loss_type=args.gan_loss_type,
            device="cuda",
        )
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    # 4.4 LPIPS & DWT
    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)

    dwt_loss = DWTLoss().cuda()

    # ================== 5. 閫夋嫨闇€瑕佷紭鍖栫殑鍙傛暟 ==================
    layers_to_opt = collect_trainable_params(net_sr, accelerator)

    # ================== 6. 鏁版嵁闆?& Dataloader ==================
    dataset_train = ExposurePairedDataset(config.train)
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    dataset_val = ExposurePairedDataset(config.validation)
    dl_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    # ================== 7. Optimizer & Scheduler ==================
    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ================== 8. Accelerator 灏佽 ==================
    net_sr, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_sr, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_de, net_lpips = accelerator.prepare(net_de, net_lpips)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_sr.to(accelerator.device, dtype=weight_dtype)
    net_de.to(accelerator.device, dtype=weight_dtype)
    net_disc.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)

    start_step = 0
    resume_from = getattr(args, "resume_from", None) if hasattr(args, "resume_from") else None
    if (resume_from is None) or (resume_from == ""):
        resume_from = os.environ.get("RESUME_FROM", None)

    if resume_from is not None and os.path.isdir(resume_from):
        accelerator.print(f"[resume] Loading state from: {resume_from}")
        accelerator.load_state(resume_from)

        meta_fp = os.path.join(resume_from, "meta.txt")
        if os.path.exists(meta_fp):
            with open(meta_fp, "r") as f:
                start_step = int(f.read().strip())
        else:
            m = re.search(r"state_(\d+)$", resume_from)
            if m:
                start_step = int(m.group(1))
        accelerator.print(f"[resume] Resumed global_step = {start_step}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=start_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    global_step = start_step

    # ================== 10. 璁粌寰幆 ==================
    for epoch in range(args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            if global_step >= args.max_train_steps:
                break

            with accelerator.accumulate(net_sr, net_disc):
                x_src = batch["lq"]
                x_tgt = batch["gt"]
                x_ori_size_src = batch["lq"]
                B = x_src.shape[0]

                # ---- 閫€鍖栭娴?----
                with torch.no_grad():
                    deg_score = net_de(x_ori_size_src.detach()).detach()

                pos_tag_prompt = [args.pos_prompt for _ in range(B)]

                # ==================================================
                # Generator: reconstruction
                # ==================================================
                x_tgt_pred = net_sr(x_src.detach(), deg_score, pos_tag_prompt)

                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.detach().float()) * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.detach().float()).mean() * args.lambda_lpips
                loss_dwt = dwt_loss(x_tgt_pred.float(), x_tgt.detach().float())
                loss_recon = loss_l2 + loss_lpips + loss_dwt

                accelerator.backward(loss_recon)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # ==================================================
                # Generator: GAN
                # ==================================================
                x_tgt_pred = net_sr(x_src.detach(), deg_score, pos_tag_prompt)
                lossG = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan

                accelerator.backward(lossG)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                # ==================================================
                # Discriminator
                # ==================================================
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(lossD_real.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(lossD_fake.mean())
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                lossD = lossD_real + lossD_fake

            # ================== 11. Logging & Checkpoint ==================
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "loss_recon": loss_recon.detach().item(),
                        "lossG": lossG.detach().item(),
                        "lossD": lossD.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "loss_dwt": loss_dwt.detach().item(),
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % 10 == 0:
                        logger.info(f"Step: {global_step} | {logs}")

                    if global_step % 50 == 0:
                        with torch.no_grad():
                            x_src_v = x_src.cpu().detach() * 0.5 + 0.5
                            x_tgt_v = x_tgt.cpu().detach() * 0.5 + 0.5
                            x_pred_v = x_tgt_pred.cpu().detach() * 0.5 + 0.5
                            combined = torch.cat([x_src_v, x_pred_v, x_tgt_v], dim=3)
                            output_pil = transforms.ToPILImage()(combined[0])
                            outf = os.path.join(args.output_dir, f"visual/train_{global_step}.png")
                            output_pil.save(outf)

                    if global_step % 2000 == 0:
                        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
                        os.makedirs(ckpt_dir, exist_ok=True)

                        outf = os.path.join(ckpt_dir, f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_sr).save_model(outf)

                        state_dir = os.path.join(ckpt_dir, f"state_{global_step}")
                        os.makedirs(state_dir, exist_ok=True)
                        accelerator.save_state(state_dir)

                        with open(os.path.join(state_dir, "meta.txt"), "w") as f:
                            f.write(str(global_step))

                    accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
