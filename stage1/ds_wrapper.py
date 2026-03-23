"""
data format is in data_format.txt
"""
import copy
import os
from typing import Dict, List
import torch
import ujson as json
import transformers
from torch.utils.data import Dataset
from PIL import Image
from stage1.utils import _pad_sequence

IGNORE_INDEX = -100  # when label is missing, set it to -100

class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        image_folder: str | None = None,
    ) -> None:
        self.processor = processor
        self.image_folder = image_folder
        with open(data_path, "r") as f:
            self.samples: List[dict] = json.load(f)
        
    # resolve path
    def _resolve_path(self, path: str) -> str:
        if os.path.exists(path) or path.startswith(("http://", "https://")):
            return path
        if self.image_folder:
            candidate = os.path.join(self.image_folder, path)
            if os.path.exists(candidate):
                return candidate
        return path  

    # load image with path or url
    def _load_image(self, src) -> Image.Image:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, str):
            path = self._resolve_path(src)
            if path.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                resp = requests.get(path, timeout=15)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            return Image.open(path).convert("RGB")
        raise TypeError(f"不支援的圖片型態：{type(src)}")

    def _normalize_messages(self, messages: List[dict]) -> List[dict]:
        messages = copy.deepcopy(messages) # use deepcopy to prevent modifying the original data
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    for key in ("image", "path", "url"):
                        if key in item:
                            item = {**item, key: self._load_image(item[key])}
                            break
                new_content.append(item)
            msg["content"] = new_content
        return messages

    # transform messages to tensor: input_ids / labels / attention_mask / pixel_values
    def _build_sample(self, messages: List[dict]) -> Dict[str, torch.Tensor]:
        processor = self.processor
        normalized = self._normalize_messages(messages)
        encoded = processor.apply_chat_template(
            normalized,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
        )
        input_ids = encoded["input_ids"].squeeze(0).long()
        attention_mask = encoded["attention_mask"].squeeze(0).long()
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        # find assistant's response token range and fill label
        assistant_roles = {"assistant", "model"}
        for idx, msg in enumerate(normalized):
            if msg["role"] not in assistant_roles:
                continue

            # find assistant's response start position: all previous turns + generation prompt
            if idx == 0:
                start_len = 0
            else:
                prefix = processor.apply_chat_template(
                    normalized[:idx],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                start_len = prefix["input_ids"].size(1)

            # 找 assistant 回答的結束位置
            prefix_with_answer = processor.apply_chat_template(
                normalized[:idx + 1],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
            )
            end_len = prefix_with_answer["input_ids"].size(1)
            labels[start_len:end_len] = input_ids[start_len:end_len]

        # 第一個 token（BOS）不計算 loss
        if labels.numel() > 0 and labels[0].item() != IGNORE_INDEX:
            labels[0] = IGNORE_INDEX

        data = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if "pixel_values" in encoded:
            data["pixel_values"] = encoded["pixel_values"]
        if "token_type_ids" in encoded:
            data["token_type_ids"] = encoded["token_type_ids"].squeeze(0).long()

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self._build_sample(self.samples[i]["messages"])


class DataCollatorForSupervisedDataset:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [ex["input_ids"] for ex in examples]
        labels_list = [ex["labels"] for ex in examples]
        has_image = any("pixel_values" in ex for ex in examples)

        input_ids = _pad_sequence(input_ids_list, padding_value=self.pad_token_id)
        labels = _pad_sequence(labels_list, padding_value=IGNORE_INDEX)
        attention_mask = (input_ids != self.pad_token_id).long()

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if has_image:
            # pixel_values shape: [total_images, C, H, W]
            batch["pixel_values"] = torch.cat(
                [ex["pixel_values"] for ex in examples if "pixel_values" in ex], dim=0
            )
            token_type_ids_list = [
                ex.get("token_type_ids", torch.zeros_like(ex["input_ids"])) for ex in examples
            ]
            batch["token_type_ids"] = _pad_sequence(token_type_ids_list, padding_value=0)

        return batch

# Enscapulate the dataset and data_collator
def make_data_module(
    processor: transformers.ProcessorMixin,
    data_path: str,
    image_folder: str | None = None,
) -> dict:
    dataset = SupervisedDataset(
        data_path=data_path,
        processor=processor,
        image_folder=image_folder,
    )
    collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )
    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
    )