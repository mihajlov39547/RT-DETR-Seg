# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
main.py
"""
import argparse
import ast
import copy
import datetime
import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import DefaultDict, List, Callable
import torch.nn as nn
from rfdetr.models.segmentation import DETRsegm
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.benchmark import benchmark
from rfdetr.util.drop_scheduler import drop_scheduler
from rfdetr.util.files import download_file
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import ModelEma, BestMetricHolder, clean_state_dict

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

logger = getLogger(__name__)

HOSTED_MODELS = {
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
    "rf-detr-nano.pth": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small.pth": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium.pth": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
}

def download_pretrain_weights(pretrain_weights: str, redownload=False):
    if pretrain_weights in HOSTED_MODELS:
        if redownload or not os.path.exists(pretrain_weights):
            logger.info(
                f"Downloading pretrained weights for {pretrain_weights}"
            )
            download_file(
                HOSTED_MODELS[pretrain_weights],
                pretrain_weights,
            )

class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.args = args
        self.resolution = args.resolution
        self.model, _, _ = build_model(args)
        self.device = torch.device(args.device)

        def _head_out_dim(m):
            if hasattr(m, "class_embed") and hasattr(m.class_embed, "out_features"):
                return m.class_embed.out_features
            if hasattr(m, "detr") and hasattr(m.detr, "class_embed") and hasattr(m.detr.class_embed, "out_features"):
                return m.detr.class_embed.out_features
            return None

        def _target_for_reinit(m):
            if hasattr(m, "reinitialize_detection_head"):
                return m
            if hasattr(m, "detr") and hasattr(m.detr, "reinitialize_detection_head"):
                return m.detr
            return None

        # ---------- Load checkpoint (prefer frozen_weights for stage-2) ----------
        ckpt_path = getattr(args, "frozen_weights", None) or getattr(args, "pretrain_weights", None)
        freeze_intent = bool(getattr(args, "masks", False) and getattr(args, "frozen_weights", None))

        if ckpt_path is not None:
            print(f"Loading weights: {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"Failed to load weights: {e}")
                # Only auto-redownload for pretrain weights
                if ckpt_path == getattr(args, "pretrain_weights", None):
                    print("Re-downloading pretrain weights…")
                    download_pretrain_weights(args.pretrain_weights, redownload=True)
                    checkpoint = torch.load(args.pretrain_weights, map_location="cpu", weights_only=False)
                else:
                    raise

            # Optional class_names from checkpoint
            if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
                self.args.class_names = checkpoint["args"].class_names
                self.class_names = checkpoint["args"].class_names

            state = checkpoint.get("model", {})
            # Find the class head bias tensor key robustly
            cls_bias_key = next((k for k in state.keys() if k.endswith("class_embed.bias")), None)
            ckpt_out_dim = state[cls_bias_key].shape[0] if cls_bias_key is not None else None
            target_out_dim = self.args.num_classes + 1  # includes background

            # --- Head shape policy: freeze vs. non-freeze ---
            if freeze_intent:
                # We are freezing the detector: DO NOT reinit. Enforce exact class count.
                cur_out = _head_out_dim(self.model)
                if ckpt_out_dim is not None and ckpt_out_dim != target_out_dim:
                    raise ValueError(
                        f"Frozen detector head out_dim ({ckpt_out_dim}) != requested ({target_out_dim}). "
                        f"Train the detector first on the SAME class set, then freeze."
                    )
                if cur_out is not None and cur_out != target_out_dim:
                    # Model was constructed with different classes than requested; that's a config error.
                    raise ValueError(
                        f"Model head out_dim ({cur_out}) != requested ({target_out_dim}) while freezing. "
                        f"Ensure args.num_classes matches the frozen checkpoint."
                    )
            else:
                # Not freezing: optionally reinit head to checkpoint size if mismatch (for best transfer)
                cur_out = _head_out_dim(self.model)
                if ckpt_out_dim is not None and cur_out is not None and cur_out != ckpt_out_dim:
                    logger.warning(
                        f"Pretrain head dim mismatch: model={cur_out}, ckpt={ckpt_out_dim}. "
                        f"Reinitializing detection head to match checkpoint ({ckpt_out_dim-1} classes)."
                    )
                    target = _target_for_reinit(self.model)
                    if target is not None:
                        target.reinitialize_detection_head(ckpt_out_dim, include_background=False)
                    else:
                        logger.warning("Cannot reinitialize head: method not found; continuing.")

            # Exclude/modify keys if requested (Obj365→COCO etc.)
            if getattr(args, "pretrain_exclude_keys", None):
                assert isinstance(args.pretrain_exclude_keys, list)
                to_drop = []
                patterns = args.pretrain_exclude_keys
                for patt in patterns:
                    if patt.endswith("*"):
                        pref = patt[:-1]
                        to_drop.extend([k for k in state.keys() if k.startswith(pref)])
                    else:
                        # treat as exact key or prefix without star
                        to_drop.extend([k for k in state.keys() if k == patt or k.startswith(patt)])
                for k in set(to_drop):
                    state.pop(k, None)

            if getattr(args, "pretrain_keys_modify_to_load", None):
                from rfdetr.util.obj365_to_coco_model import get_coco_pretrain_from_obj365
                cur_sd = self.model.state_dict()
                assert isinstance(args.pretrain_keys_modify_to_load, list)
                for k in args.pretrain_keys_modify_to_load:
                    try:
                        state[k] = get_coco_pretrain_from_obj365(cur_sd[k], state[k])
                    except Exception:
                        print(f"Failed to adapt {k}, removing from checkpoint")
                        state.pop(k, None)

            # Resize query params if group_detr differs
            num_desired_queries = args.num_queries * args.group_detr
            for name in list(state.keys()):
                if name.endswith("refpoint_embed.weight") or name.endswith("query_feat.weight"):
                    state[name] = state[name][:num_desired_queries]

            # Finally load
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:   print(f"[load_state_dict] missing: {len(missing)} params")
            if unexpected:print(f"[load_state_dict] unexpected: {len(unexpected)} params")

        # ---------- Optional: apply LoRA to backbone ----------
        if getattr(args, "backbone_lora", False):
            print("Applying LORA to backbone")
            lora_config = LoraConfig(
                r=16, lora_alpha=16, use_dora=True,
                target_modules=[
                    "q_proj", "v_proj", "k_proj",          # OWL-ViT
                    "qkv",                                 # OpenCLIP/SigLIP2
                    "query", "key", "value", "cls_token", "register_tokens",  # DINOv2 windowed
                ]
            )
            self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)

        # ---------- Move to device & build losses/postprocs ----------
        self.model = self.model.to(self.device)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(args)
        self.stop_early = False

    
    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def train(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
        tensorboard_writer = kwargs.pop("tensorboard_writer", None)
        # helper to strip unpicklable things from args before saving
        def _safe_args(ns):
            try:
                d = vars(ns).copy()
            except Exception:
                return {}
            d.pop("tensorboard_writer", None)
            return d
        currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]
        for key in callbacks.keys():
            if key not in currently_supported_callbacks:
                raise ValueError(
                    f"Callback {key} is not currently supported, please file an issue if you need it!\n"
                    f"Currently supported callbacks: {currently_supported_callbacks}"
                )
        args = populate_args(**kwargs)
        freeze_intent = bool(getattr(args, "masks", False) and getattr(args, "frozen_weights", None))
        if freeze_intent:
            print("[Stage-2] Freezing detector; training mask head only.")
            # Optional default: if user didn’t override LR, make it small for the tiny seg head
            if not hasattr(args, "lr_overridden") and "lr" not in kwargs:
                args.lr = min(args.lr, 5e-5)  # gentle default; user can still pass --lr to override

        
            self.args.class_names = args.class_names
            self.args.num_classes = args.num_classes

        utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(utils.get_sha()))
        print(args)
        device = torch.device(args.device)
        
        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Decide freeze intent once
        freeze_intent = bool(getattr(args, "masks", False) and getattr(args, "frozen_weights", None))

        # Keep user/configured aux_loss as-is (needed for seg to learn across decoder layers)

        model = self.model
        model.to(device)

        # Wrap with DETRsegm if masks are enabled
        if getattr(args, "masks", False) and not isinstance(model, DETRsegm):
            print("[fixup] Wrapping detector with DETRsegm (masks=True).")
            model = DETRsegm(model, freeze_detr=freeze_intent).to(device)

        # Keep model.detr.aux_loss unchanged; Stage-2 relies on aux supervision

        model_without_ddp = model

        # Build criterion/postprocessors AFTER aux is disabled
        criterion, postprocessors = build_criterion_and_postprocessors(args)
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        param_dicts = get_param_dict(args, model_without_ddp)

        # --- Normalize param groups so 'params' is always a list of nn.Parameter ---
        filtered_groups = []
        for g in param_dicts:
            raw = g["params"]

            # Make sure we have a list of Parameters (not a single Tensor / generator / dict_values)
            if isinstance(raw, (torch.nn.Parameter, torch.Tensor)):
                raw = [raw]
            else:
                raw = list(raw)  # handle generators/iterables

            # Keep only real leaf Parameters that require grad
            params = [p for p in raw if isinstance(p, torch.nn.Parameter) and p.requires_grad]
            if not params:
                continue

            gg = {k: v for k, v in g.items() if k != "params"}
            gg["params"] = params
            filtered_groups.append(gg)

        param_dicts = filtered_groups

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        # Choose the learning rate scheduler based on the new argument

        dataset_train = build_dataset(image_set='train', args=args, resolution=args.resolution)
        dataset_val = build_dataset(image_set='val', args=args, resolution=args.resolution)
        dataset_test = build_dataset(image_set='test', args=args, resolution=args.resolution)

        # for cosine annealing, calculate total training steps and warmup steps
        total_batch_size_for_lr = args.batch_size * utils.get_world_size() * args.grad_accum_steps
        num_training_steps_per_epoch_lr = (len(dataset_train) + total_batch_size_for_lr - 1) // total_batch_size_for_lr
        total_training_steps_lr = num_training_steps_per_epoch_lr * args.epochs
        warmup_steps_lr = num_training_steps_per_epoch_lr * args.warmup_epochs
        def lr_lambda(current_step: int):
            if current_step < warmup_steps_lr:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps_lr))
            else:
                # Cosine annealing from multiplier 1.0 down to lr_min_factor
                if args.lr_scheduler == 'cosine':
                    progress = float(current_step - warmup_steps_lr) / float(max(1, total_training_steps_lr - warmup_steps_lr))
                    return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
                elif args.lr_scheduler == 'step':
                    if current_step < args.lr_drop * num_training_steps_per_epoch_lr:
                        return 1.0
                    else:
                        return 0.1
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        effective_batch_size = args.batch_size * args.grad_accum_steps
        min_batches = kwargs.get('min_batches', 5)
        if len(dataset_train) < effective_batch_size * min_batches:
            logger.info(
                f"Training with uniform sampler because dataset is too small: {len(dataset_train)} < {effective_batch_size * min_batches}"
            )
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=effective_batch_size * min_batches,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_size=effective_batch_size,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                sampler=sampler,
            )
            model_without_ddp.train()
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, effective_batch_size, drop_last=True)
            data_loader_train = DataLoader(
                dataset_train, 
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn, 
                num_workers=args.num_workers
            )
            # --- Sanity: one forward to verify outputs ---
            was_training = model_without_ddp.training
            model_without_ddp.eval()
            with torch.no_grad():
                samples, _ = next(iter(data_loader_train))  # (NestedTensor OR list[tensor], list[dict])
                samples = samples.to(device) if hasattr(samples, "to") else [s.to(device) for s in samples]
                out = model_without_ddp(samples, None)
                print("[sanity] keys:", list(out.keys()))
                print("[sanity] has pred_masks:", 'pred_masks' in out, "| has aux_outputs:", 'aux_outputs' in out)
            if was_training:
                model_without_ddp.train()
        
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, 
                                    num_workers=args.num_workers)
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                    drop_last=False, collate_fn=utils.collate_fn, 
                                    num_workers=args.num_workers)

        base_ds = get_coco_api_from_dataset(dataset_val)
        base_ds_test = get_coco_api_from_dataset(dataset_test)
        if args.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None


        output_dir = Path(args.output_dir)
        
        if  utils.is_main_process():
            print("Get benchmark")
            if args.do_benchmark:
                benchmark_model = copy.deepcopy(model_without_ddp)
                bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
                print(json.dumps(bm, indent=2))
                del benchmark_model
        
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            if args.use_ema:
                if 'ema_model' in checkpoint:
                    self.ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
                else:
                    del self.ema_m
                    self.ema_m = ModelEma(model, decay=args.ema_decay, tau=args.ema_tau) 
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:                
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        if args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args=args, tensorboard_writer=tensorboard_writer, epoch=None)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
                if "segm" in coco_evaluator.coco_eval:
                    utils.save_on_master(coco_evaluator.coco_eval["segm"].eval, output_dir / "eval_masks.pth")
            return
        
        # for drop
        total_batch_size = effective_batch_size * utils.get_world_size()
        num_training_steps_per_epoch = (len(dataset_train) + total_batch_size - 1) // total_batch_size
        schedules = {}
        if args.dropout > 0:
            schedules['do'] = drop_scheduler(
                args.dropout, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

        if args.drop_path > 0:
            schedules['dp'] = drop_scheduler(
                args.drop_path, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))

        print("Start training")
        start_time = time.time()
        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        best_map_5095 = 0
        best_map_50 = 0
        best_map_ema_5095 = 0
        best_map_ema_50 = 0
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)

            model.train()
            criterion.train()
            train_stats = train_one_epoch(
                model, criterion, lr_scheduler, data_loader_train, optimizer, device, epoch,
                effective_batch_size, args.clip_max_norm, ema_m=self.ema_m, schedules=schedules, 
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers, args=args, callbacks=callbacks, tensorboard_writer=tensorboard_writer)
            train_epoch_time = time.time() - epoch_start_time
            train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # add an extra dated checkpoint at milestones
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': _safe_args(args),
                    }
                    if args.use_ema:
                        weights['ema_model'] = self.ema_m.module.state_dict()
                    if not args.dont_save_weights:
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        utils.save_on_master(weights, checkpoint_path)

            with torch.inference_mode():
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, args=args, tensorboard_writer=tensorboard_writer, epoch=epoch
                )
            map_regular = test_stats["coco_eval_bbox"][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                best_map_5095 = max(best_map_5095, map_regular)
                best_map_50 = max(best_map_50, test_stats["coco_eval_bbox"][1])
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                if not args.dont_save_weights:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': _safe_args(args),
                    }, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            if "coco_eval_masks" in test_stats:
                log_stats["test_mask_AP@[.5:.95]"] = float(test_stats["coco_eval_masks"][0])
                log_stats["test_mask_AP@.50"]      = float(test_stats["coco_eval_masks"][1])
            if args.use_ema:
                ema_test_stats, _ = evaluate(
                    self.ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device,
                    args=args, tensorboard_writer=tensorboard_writer, epoch=epoch
                )
                log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
                map_ema = ema_test_stats["coco_eval_bbox"][0]
                best_map_ema_5095 = max(best_map_ema_5095, map_ema)
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
                if _isbest:
                    best_map_ema_50 = max(best_map_ema_50, ema_test_stats["coco_eval_bbox"][1])
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    if not args.dont_save_weights:
                        utils.save_on_master({
                            'model': self.ema_m.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': _safe_args(args),
                        }, checkpoint_path)
            log_stats.update(best_map_holder.summary())
            
            # epoch parameters
            ep_paras = {
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            log_stats['train_epoch_time'] = train_epoch_time_str
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
                        if "segm" in coco_evaluator.coco_eval:
                            filenames = ['latest_masks.pth']
                            if epoch % 50 == 0:
                                filenames.append(f'masks_{epoch:03}.pth')
                            for name in filenames:
                                torch.save(coco_evaluator.coco_eval["segm"].eval,
                                        output_dir / "eval" / name)
            
            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)

            if self.stop_early:
                print(f"Early stopping requested, stopping at epoch {epoch}")
                break

        best_is_ema = best_map_ema_5095 > best_map_5095
        
        if utils.is_main_process():
            if best_is_ema:
                shutil.copy2(output_dir / 'checkpoint_best_ema.pth', output_dir / 'checkpoint_best_total.pth')
            else:
                shutil.copy2(output_dir / 'checkpoint_best_regular.pth', output_dir / 'checkpoint_best_total.pth')
            
            utils.strip_checkpoint(output_dir / 'checkpoint_best_total.pth')
        
            best_map_5095 = max(best_map_5095, best_map_ema_5095)
            if best_is_ema:
                results = ema_test_stats["results_json"]
                if "results_json_masks" in ema_test_stats:
                    results["masks"] = ema_test_stats["results_json_masks"]
            else:
                results = test_stats["results_json"]
                if "results_json_masks" in test_stats:
                    results["masks"] = test_stats["results_json_masks"]


            class_map = results["class_map"]
            results["class_map"] = {"valid": class_map}
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            print('Results saved to {}'.format(output_dir / "results.json"))
            
        
        if best_is_ema:
            self.model = self.ema_m.module
        self.model.eval()


        if args.run_test:
            best_state_dict = torch.load(output_dir / 'checkpoint_best_total.pth',
                                        map_location='cpu', weights_only=False)['model']
            model.load_state_dict(best_state_dict)
            model.eval()

            test_stats, _ = evaluate(
                model, criterion, postprocessors, data_loader_test, base_ds_test, device, args=args
            )
            print(f"Test results: {test_stats}")

            # Load existing results.json (created earlier at end of training)
            with open(output_dir / "results.json", "r") as f:
                results = json.load(f)

            # ---- Add bbox test metrics ----
            # Ensure structure exists
            results.setdefault("class_map", {})
            # Per-class table for test set
            results["class_map"]["test"] = test_stats["results_json"]["class_map"]

            # ---- Add mask test metrics (if available) ----
            if "results_json_masks" in test_stats:
                results.setdefault("masks", {})
                # Keep the same structure as 'valid': a top-level summary + per-class table nested under 'class_map'
                results["masks"].setdefault("class_map", {})
                results["masks"]["class_map"]["test"] = test_stats["results_json_masks"]["class_map"]
                # Optional: summary numbers for test
                results["masks"]["map_test"] = test_stats["results_json_masks"]["map"]
                results["masks"]["precision_test"] = test_stats["results_json_masks"]["precision"]
                results["masks"]["recall_test"] = test_stats["results_json_masks"]["recall"]

            # Single write after all updates
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f, indent=2)


        for callback in callbacks["on_train_end"]:
            callback()
    
    def export(self, output_dir="output", infer_dir=None, simplify=False,  backbone_only=False, opset_version=17, verbose=True, force=False, shape=None, batch_size=1, **kwargs):
        """Export the trained model to ONNX format"""
        print(f"Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import export_onnx, onnx_simplify, make_infer_image
        except ImportError:
            print("It seems some dependencies for ONNX export are missing. Please run `pip install rfdetr[onnxexport]` and try again.")
            raise


        device = self.device
        model = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            ps, nw = self.model.args.patch_size, self.model.args.num_windows
            need = ps * nw
            if (shape[0] % need) != 0 or (shape[1] % need) != 0:
                raise ValueError(f"Shape must be divisible by patch_size*num_windows ({need})")


        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names = ['input']

        # --- Run a quick forward to decide outputs (and to sanity-check shapes) ---
        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
                output_names = ['features']
                has_masks = False
            else:
                outputs = model(input_tensors)
                dets = outputs['pred_boxes']
                labels = outputs['pred_logits']
                has_masks = ('pred_masks' in outputs)
                if has_masks:
                    masks = outputs['pred_masks']
                    print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
                else:
                    print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
                # ONNX prefers tuple/list outputs; wrap the dict → tuple
                class _ExportDetrWrapper(nn.Module):
                    def __init__(self, m, return_masks=False):
                        super().__init__()
                        self.m = m
                        self.return_masks = return_masks
                    def forward(self, x):
                        out = self.m(x)
                        if self.return_masks and 'pred_masks' in out:
                            return out['pred_boxes'], out['pred_logits'], out['pred_masks']
                        return out['pred_boxes'], out['pred_logits']
                model = _ExportDetrWrapper(model, return_masks=has_masks)
                output_names = ['dets', 'labels'] + (['masks'] if has_masks else [])

        model.cpu()
        input_tensors = input_tensors.cpu()
        dynamic_axes = None  # keep as-is; add if you want variable batch/seq dims

        # Export to ONNX
        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version
        )

        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force
            )
            print(f"Successfully simplified ONNX model to: {sim_output_file}")

        print("ONNX export completed successfully")
        self.model = self.model.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    config = vars(args)  # Convert Namespace to dictionary
    
    if args.subcommand == 'export_model':
        filter_keys = [
            "num_classes",
            "grad_accum_steps",
            "lr",
            "lr_encoder",
            "weight_decay",
            "epochs",
            "lr_drop",
            "clip_max_norm",
            "lr_vit_layer_decay",
            "lr_component_decay",
            "dropout",
            "drop_path",
            "drop_mode",
            "drop_schedule",
            "cutoff_epoch",
            "pretrained_encoder",
            "pretrain_weights",
            "pretrain_exclude_keys",
            "pretrain_keys_modify_to_load",
            "freeze_florence",
            "freeze_aimv2",
            "decoder_norm",
            "set_cost_class",
            "set_cost_bbox",
            "set_cost_giou",
            "cls_loss_coef",
            "bbox_loss_coef",
            "giou_loss_coef",
            "focal_alpha",
            "aux_loss",
            "sum_group_losses",
            "use_varifocal_loss",
            "use_position_supervised_loss",
            "ia_bce_loss",
            "dataset_file",
            "coco_path",
            "dataset_dir",
            "square_resize_div_64",
            "output_dir",
            "checkpoint_interval",
            "seed",
            "resume",
            "start_epoch",
            "eval",
            "use_ema",
            "ema_decay",
            "ema_tau",
            "num_workers",
            "device",
            "world_size",
            "dist_url",
            "sync_bn",
            "fp16_eval",
            "infer_dir",
            "verbose",
            "opset_version",
            "dry_run",
            "shape",
        ]
        for key in filter_keys:
            config.pop(key, None)  # Use pop with None to avoid KeyError
            
        from rfdetr.deploy.export import main as export_main
        if args.batch_size != 1:
            config['batch_size'] = 1
            print(f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1.")
        export_main(**config)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--grad_accum_steps', default=1, type=int)
    parser.add_argument('--amp', default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)
    parser.add_argument('--do_benchmark', action='store_true', help='benchmark the model')

    # drop args 
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # early / late dropout and stochastic depth settings
    parser.add_argument('--drop_mode', type=str, default='standard',
                        choices=['standard', 'early', 'late'], help='drop mode')
    parser.add_argument('--drop_schedule', type=str, default='constant',
                        choices=['constant', 'linear'],
                        help='drop schedule for early dropout / s.d. only')
    parser.add_argument('--cutoff_epoch', type=int, default=0,
                        help='if drop_mode is early / late, this is the epoch where dropout ends / starts')

    # Model parameters
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    parser.add_argument('--pretrain_weights', type=str, default=None, 
                        help="Path to the pretrained model.")
    parser.add_argument('--pretrain_exclude_keys', type=str, default=None, nargs='+', 
                        help="Keys you do not want to load.")
    parser.add_argument('--pretrain_keys_modify_to_load', type=str, default=None, nargs='+',
                        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.")

    # * Backbone
    parser.add_argument('--encoder', default='vit_tiny', type=str,
                        help="Name of the transformer or convolutional encoder to use")
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int,
                        help="Number of layers used in ViT encoder")
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+', help='only for vit now')
    parser.add_argument("--freeze_encoder", action="store_true", dest="freeze_encoder")
    parser.add_argument("--layer_norm", action="store_true", dest="layer_norm")
    parser.add_argument("--rms_norm", action="store_true", dest="rms_norm")
    parser.add_argument("--backbone_lora", action="store_true", dest="backbone_lora")
    parser.add_argument("--force_no_pretrain", action="store_true", dest="force_no_pretrain")

    # * Transformer
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--sa_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's self-attentions")
    parser.add_argument('--ca_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's cross-attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--group_detr', default=13, type=int,
                        help="Number of groups to speed up detr training")
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true', help='lite refpoint refine mode for speed-up')
    parser.add_argument('--num_select', default=100, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='the number of sampling points')
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')
    parser.add_argument('--freeze_batch_norm', action='store_true')
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--square_resize_div_64', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--dont_save_weights', action='store_true')
    parser.add_argument('--checkpoint_interval', default=1, type=int,
                        help='epoch interval to save checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.9997, type=float)
    parser.add_argument('--ema_tau', default=0, type=float)

    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--sync_bn', default=True, type=bool,
                        help='setup synchronized BatchNorm for distributed training')
    
    # fp16
    parser.add_argument('--fp16_eval', default=False, action='store_true',
                        help='evaluate in fp16 precision.')

    # custom args
    parser.add_argument('--encoder_only', action='store_true', help='Export and benchmark encoder only')
    parser.add_argument('--backbone_only', action='store_true', help='Export and benchmark backbone only')
    parser.add_argument('--resolution', type=int, default=640, help="input resolution")
    parser.add_argument('--use_cls_token', action='store_true', help='use cls token')
    parser.add_argument('--multi_scale', action='store_true', help='use multi scale')
    parser.add_argument('--expanded_scales', action='store_true', help='widen multi-scale set')
    parser.add_argument('--do_random_resize_via_padding', action='store_true', help='use random resize via padding')
    parser.add_argument('--warmup_epochs', default=1, type=float, 
        help='Number of warmup epochs for linear warmup before cosine annealing')
    # Add scheduler type argument: 'step' or 'cosine'
    parser.add_argument(
        '--lr_scheduler',
        default='step',
        choices=['step', 'cosine'],
        help="Type of learning rate scheduler to use: 'step' (default) or 'cosine'"
    )
    parser.add_argument('--lr_min_factor', default=0.0, type=float, 
        help='Minimum learning rate factor (as a fraction of initial lr) at the end of cosine annealing')
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping based on mAP improvement')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--early_stopping_min_delta', default=0.001, type=float,
                        help='Minimum change in mAP to qualify as an improvement')
    parser.add_argument('--early_stopping_use_ema', action='store_true',
                        help='Use EMA model metrics for early stopping')
    # subparsers
    subparsers = parser.add_subparsers(title='sub-commands', dest='subcommand',
        description='valid subcommands', help='additional help')

    # subparser for export model
    parser_export = subparsers.add_parser('export_model', help='LWDETR model export')
    parser_export.add_argument('--infer_dir', type=str, default=None)
    parser_export.add_argument('--verbose', type=ast.literal_eval, default=False, nargs="?", const=True)
    parser_export.add_argument('--opset_version', type=int, default=17)
    parser_export.add_argument('--simplify', action='store_true', help="Simplify onnx model")
    parser_export.add_argument('--tensorrt', '--trtexec', '--trt', action='store_true',
                               help="build tensorrt engine")
    parser_export.add_argument('--dry-run', '--test', '-t', action='store_true', help="just print command")
    parser_export.add_argument('--profile', action='store_true', help='Run nsys profiling during TensorRT export')
    parser_export.add_argument('--shape', type=int, nargs=2, default=(640, 640), help="input shape (width, height)")

    # Segmentation args
    parser.add_argument('--masks', action='store_true', help='enable instance segmentation head')
    parser.add_argument('--mask_loss_coef', type=float, default=1.0)
    parser.add_argument('--dice_loss_coef', type=float, default=1.0)
    parser.add_argument('--eos_coef', type=float, default=0.1)
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help='freeze detector when wrapping with DETRsegm if a path is provided')


    return parser

def populate_args(
    # Basic training parameters
    num_classes=2,
    grad_accum_steps=1,
    amp=False,
    lr=1e-4,
    lr_encoder=1.5e-4,
    batch_size=2,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    do_benchmark=False,
    
    # Drop parameters
    dropout=0,
    drop_path=0,
    drop_mode='standard',
    drop_schedule='constant',
    cutoff_epoch=0,
    
    # Model parameters
    pretrained_encoder=None,
    pretrain_weights=None, 
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,
    
    # Backbone parameters
    encoder='vit_tiny',
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding='sine',
    out_feature_indexes=[-1],
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    
    # Transformer parameters
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale='P4',
    lite_refpoint_refine=False,
    num_select=100,
    dec_n_points=4,
    decoder_norm='LN',
    bbox_reparam=False,
    freeze_batch_norm=False,
    
    # Matcher parameters
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    
    # Loss coefficients
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,
    
    # Dataset parameters
    dataset_file='coco',
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,
    
    # Output parameters
    output_dir='output',
    dont_save_weights=False,
    checkpoint_interval=1,
    seed=42,
    resume='',
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=2,
    
    # Distributed training parameters
    device='cuda',
    world_size=1,
    dist_url='env://',
    sync_bn=True,
    
    # FP16
    fp16_eval=False,
    
    # Custom args
    encoder_only=False,
    backbone_only=False,
    resolution=640,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    warmup_epochs=1,
    lr_scheduler='step',
    lr_min_factor=0.0,
    # Early stopping parameters
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    gradient_checkpointing=False,
    # Segmentation parameters
    masks=False,
    mask_loss_coef=1.0,
    dice_loss_coef=1.0,
    eos_coef=0.1,
    frozen_weights=None,
    # Additional
    subcommand=None,
    **extra_kwargs  # To handle any unexpected arguments
):
    extra_kwargs.pop("tensorboard_writer", None)
    args = argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        lite_refpoint_refine=lite_refpoint_refine,
        num_select=num_select,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=do_random_resize_via_padding,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        gradient_checkpointing=gradient_checkpointing,
        masks=masks,
        mask_loss_coef=mask_loss_coef,
        dice_loss_coef=dice_loss_coef,
        eos_coef=eos_coef,
        frozen_weights=frozen_weights,
        **extra_kwargs
    )
    return args