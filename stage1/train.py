"""
uv run deepspeed stage1/main.py --deepspeed scripts/stage1.json [args...]
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
)

from stage1.ds_wrapper import make_data_module
from stage1.forward import replace_forward
from stage1.sft import GemmaSFTTrainer
from stage1.utils import _freeze_llm, _unfreeze_vision, _print_trainable_parameters, _log


@dataclass
class ModelArguments:
    model_id: str = field(
        default="google/gemma-3-4b-it",
        metadata={"help": "HuggingFace model ID or local model path"},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "training data json file path"}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "image root directory"},
    )


@dataclass
class Stage1TrainingArguments(TrainingArguments):
    vision_lr: Optional[float] = field(
        default=2e-5,
        metadata={"help": "Vision Tower learning rate"},
    )
    projector_lr: Optional[float] = field(
        default=2e-5,
        metadata={"help": "Projector learning rate"},
    )

    max_seq_length: int = field(
        default=4096,
        metadata={"help": "max sequence length"},
    )

    # Stage 1 固定不變的設定（由程式硬鎖，不接受 CLI 覆蓋）
    cache_dir: Optional[str] = field(default=None)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, Stage1TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = torch.bfloat16
    device = training_args.device
    replace_forward() # replace the forward function of Gemma3ForConditionalGeneration
    _log(f"Loading model: {model_args.model_id}")

    # wake up gemma3-4b-it with bfloat16
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        cache_dir=training_args.cache_dir,
        attn_implementation="eager",  # use eager instead of flash attention first to prevent tensor mismatch
    )

    # freeze lm backbone, unfreeze vision components
    _freeze_llm(model)
    _unfreeze_vision(model, compute_dtype, device)

    # cache gradient checkpoint and reused it(which can save VRAM)
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # model config
    model.config.use_cache = False # close kv-cache to purely train 
    model.config.vision_lr = training_args.vision_lr
    model.config.projector_lr = training_args.projector_lr

    # processor & ds wrapper
    # specific processor developed by google for wrapping image and text
    processor = AutoProcessor.from_pretrained(model_args.model_id)
    data_module = make_data_module(
        processor=processor,
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
    )

    # trainer
    trainer = GemmaSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    # store training progress or continue training
    output_dir = pathlib.Path(training_args.output_dir)
    resume = bool(list(output_dir.glob("checkpoint-*")))
    _log(f"{'continue training' if resume else 'new training'} → {training_args.output_dir}")
    trainer.train(resume_from_checkpoint=resume)

    # save model
    trainer.save_state()
    model.config.use_cache = True # reopen kv-cache for application usage

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(training_args.output_dir)
    else:
        state_dict = {k: v.cpu() for k, v in trainer.model.state_dict().items()}
        trainer._save(training_args.output_dir, state_dict=state_dict)
        trainer.model.config.save_pretrained(training_args.output_dir)

    _log("Training completed, model saved to", training_args.output_dir)


if __name__ == "__main__":
    train()