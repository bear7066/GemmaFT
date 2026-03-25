"""
1. Create optimizer for  image_encoder(vision_tower), projector(multi_modal_projector)
2. Set up different learning rate for image_encoder, projector 
"""

import os
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
)


class GemmaSFTTrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model
        if self.optimizer is not None:
            return self.optimizer

        # exclude parameters that don't need L2 normalization(prevent gradient explode), like bias and layernorm
        decay_params = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        # config learning rate from run.sh
        lr_mapper: dict[str, float] = {}
        if self.args.projector_lr is not None:
            lr_mapper["multi_modal_projector"] = self.args.projector_lr
        if self.args.image_encoder_lr is not None:
            lr_mapper["vision_tower"] = self.args.image_encoder_lr

        # get the actual name needed to be set different learning rate
        special_names: set[str] = set()
        for keyword in lr_mapper:
            for n, _ in opt_model.named_parameters():
                if keyword in n:
                    special_names.add(n)

        # pack up normal parameters
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n in decay_params and n not in special_names and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n not in decay_params and n not in special_names and p.requires_grad # use requires_grad to make sure freezing
                ],
                "weight_decay": 0.0,
            },
        ]

        # pack up special parameters
        for keyword, lr in lr_mapper.items():
            module_names = {n for n, _ in opt_model.named_parameters() if keyword in n}
            optimizer_grouped_parameters.extend([
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n in decay_params and n in module_names and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if n not in decay_params and n in module_names and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ])

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch,
            ignore_keys_for_eval, start_time, learning_rate=learning_rate,
        )

        if self.control.should_log and self.optimizer is not None:
            logs = {}
            for i, pg in enumerate(self.optimizer.param_groups):
                name = pg.get("param_group_name", f"group_{i}")
                logs[f"lr_{name}"] = pg["lr"]
            self.log(logs)