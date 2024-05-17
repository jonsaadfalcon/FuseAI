"""2. Tokenize and then forward with all models for their logits."""

import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import DatasetDict, Features, load_dataset, load_from_disk
#from src.utils.common import load_tokenizer_and_model
#from src.utils.data_collator import DataCollatorForSeq2Seq
#from src.utils.others import (
#    IGNORE_TOKEN_ID,
#    AttrDict,
#    dict_to_list,
#    get_logger,
#    get_tokenizer,
#    release_model_and_tensor,
#)





########################################################

"""All the config."""

from dataclasses import dataclass, field
from typing import Optional

from transformers import Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "The max train samples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "The max eval samples."}
    )
    max_predict_samples: Optional[int] = field(
        default=None, metadata={"help": "The max predict samples."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=64,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    # common args
    training_mode: Optional[str] = field(
        default="full", metadata={"help": "The training mode: full or qlora."}
    )
    use_flash_attn: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use flash attention."}
    )
    cache_dir: Optional[str] = field(default=None)
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": "Number of subprocesses to use for data loading (PyTorch only). "
            "0 means that the data will be loaded in the main process."
        },
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    optim: str = field(
        default="adamw_torch", metadata={"help": "adamw_torch or paged_adamw_32bit."}
    )
    # qlora args
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    max_memory_MB: int = field(default=40000, metadata={"help": "Free memory per gpu."})
    # distill args
    do_distill: Optional[bool] = field(
        default=False, metadata={"help": "Whether to distill logits during training."}
    )
    distill_with_ref_model: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use ref model during distilling."}
    )
    distill_with_aligned_model_0: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use aligned model 0 duriing distilling."},
    )
    distill_with_aligned_model_1: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use aligned model 1 duriing distilling."},
    )
    distill_loss_type: Optional[str] = field(
        default="ce", metadata={"help": "The distill loss type, could be ce or kl."}
    )
    distill_teacher_temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The temperature used for teacher during distilling."},
    )
    lm_loss_weight: Optional[float] = field(
        default=1.0, metadata={"help": "The weight of language loss during distilling."}
    )
    distill_greater_as_gt: Optional[bool] = field(
        default=False,
        metadata={"help": "Use logits from greater teacher as ground truth label."},
    )
    distill_greater_as_gt_type: Optional[str] = field(
        default="hard", metadata={"help": "hard or hard_and_decay or soft."}
    )
    distill_weighted_as_gt: Optional[bool] = field(
        default=False,
        metadata={"help": "Use logits from weighted teacher as ground truth label."},
    )
    distill_weighted_as_gt_type: Optional[str] = field(
        default="hard", metadata={"help": "hard or hard_and_decay or soft."}
    )


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_length: Optional[int] = field(default=4096)
    max_new_tokens: Optional[int] = field(default=None)
    min_new_tokens: Optional[int] = field(default=None)

    # Generation strategy
    do_sample: Optional[bool] = field(default=True)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=0.6)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=0.9)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


"""Prepare args, tokenizer, model."""

import argparse

import transformers


def prepare_args():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    model_args, data_args, training_args, generation_args, extra_args = (
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    )
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    if args.training_mode == "full":
        args.optim = "adamw_torch"
    elif args.training_mode == "qlora":
        args.optim = "paged_adamw_32bit"
    else:
        logger.warning(f"Now {args.training_mode} is not supported.")
        raise NotImplementedError
    logger.info(f"Training/Evaluation Args: {args}")
    return model_args, data_args, training_args, args


def load_tokenizer_and_model(args):
    tokenizer, kwargs = get_tokenizer(
        args.model_name_or_path, args.cache_dir, args.model_max_length
    )
    if args.training_mode == "full":
        model = get_base_model(
            args, trust_remote_code=kwargs["model_trust_remote_code"]
        )
    elif args.training_mode == "qlora":
        checkpoint_dir, completed_training = get_last_checkpoint_for_lora(
            args.output_dir
        )
        model = get_accelerate_model(
            args,
            checkpoint_dir=checkpoint_dir,
            trust_remote_code=kwargs["model_trust_remote_code"],
        )
    else:
        logger.warning(f"Now {args.training_mode} is not supported.")
        raise NotImplementedError
    model.config.use_cache = False
    return tokenizer, model


# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import Seq2SeqTrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = self.max_length
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)
        delete_keys = dict()  # save other keys
        for k in features[0].keys():
            if k not in ["input_ids", "attention_mask", "labels"]:
                delete_keys[k] = []
        if len(delete_keys.keys()) > 0:
            for feature in features:
                for k in delete_keys.keys():
                    delete_keys[k].append(feature[k])
                    del feature[k]
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if len(delete_keys.keys()) > 0:
            for k, v in delete_keys.items():
                features[k] = v

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features


@dataclass
class DataCollatorForDistill:
    """
    Data collator that will dynamically pad the inputs and labels, then weighted sum and pad all logits.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
        training_args: Seq2SeqTrainingArguments = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.training_args = training_args
        self.vocab = self.tokenizer.get_vocab().keys()
        self.pad_id = self.tokenizer.pad_token_id
        self.end_id = self.tokenizer.eos_token_id

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = self.max_length
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)
        delete_keys = dict()  # save other keys
        for k in features[0].keys():
            if k not in ["input_ids", "attention_mask", "labels"]:
                delete_keys[k] = []
        if len(delete_keys.keys()) > 0:
            for feature in features:
                for k in delete_keys.keys():
                    delete_keys[k].append(feature[k])
                    del feature[k]
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        if len(delete_keys.keys()) > 0:
            for k, v in delete_keys.items():
                features[k] = v
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        # weighted sum and pad all logits
        batch_size = features["input_ids"].size(0)
        vocab_size = len(self.vocab)
        base_target_dist = torch.zeros(batch_size, self.max_length, vocab_size).to(
            torch.bfloat16
        )
        aligned_target_dist_0 = (
            torch.zeros(batch_size, self.max_length, vocab_size).to(torch.bfloat16)
            if (
                self.training_args.distill_with_aligned_model_0 is True
                and "per_step_aligned_logits_0" in features
            )
            else None
        )
        aligned_target_dist_1 = (
            torch.zeros(batch_size, self.max_length, vocab_size).to(torch.bfloat16)
            if (
                self.training_args.distill_with_aligned_model_1 is True
                and "per_step_aligned_logits_1" in features
            )
            else None
        )
        for i in range(batch_size):
            base_seq_len = len(features["per_step_logits"][i])
            for j in range(self.max_length):
                if j < base_seq_len:
                    base_logits = torch.tensor(
                        features["per_step_logits"][i][j], dtype=torch.bfloat16
                    )
                    base_prob = softmax(
                        base_logits / self.training_args.distill_teacher_temperature, -1
                    )
                    base_indices = torch.tensor(features["per_step_indices"][i][j])
                    base_target_dist[i][j] = base_target_dist[i][j].scatter_(
                        -1, base_indices, base_prob
                    )

                    if (
                        aligned_target_dist_0 is not None
                        and len(features["per_step_aligned_indices_0"][i][j]) > 0
                    ):
                        aligned_logits_0 = torch.tensor(
                            features["per_step_aligned_logits_0"][i][j],
                            dtype=torch.bfloat16,
                        )
                        aligned_prob_0 = softmax(
                            aligned_logits_0
                            / self.training_args.distill_teacher_temperature,
                            -1,
                        )
                        aligned_indices_0 = torch.tensor(
                            features["per_step_aligned_indices_0"][i][j]
                        )
                        aligned_target_dist_0[i][j] = aligned_target_dist_0[i][
                            j
                        ].scatter_(-1, aligned_indices_0, aligned_prob_0)
                    elif aligned_target_dist_0 is not None:
                        aligned_target_dist_0[i][j] = base_target_dist[i][j]  # bad case

                    if (
                        aligned_target_dist_1 is not None
                        and len(features["per_step_aligned_indices_1"][i][j]) > 0
                    ):
                        aligned_logits_1 = torch.tensor(
                            features["per_step_aligned_logits_1"][i][j],
                            dtype=torch.bfloat16,
                        )
                        aligned_prob_1 = softmax(
                            aligned_logits_1
                            / self.training_args.distill_teacher_temperature,
                            -1,
                        )
                        aligned_indices_1 = torch.tensor(
                            features["per_step_aligned_indices_1"][i][j]
                        )
                        aligned_target_dist_1[i][j] = aligned_target_dist_1[i][
                            j
                        ].scatter_(-1, aligned_indices_1, aligned_prob_1)
                    elif aligned_target_dist_1 is not None:
                        aligned_target_dist_1[i][j] = base_target_dist[i][j]  # bad case

                else:  # padding position
                    base_target_dist[i][j][self.pad_id] = 1.0
                    if aligned_target_dist_0 is not None:
                        aligned_target_dist_0[i][j][self.pad_id] = 1.0
                    if aligned_target_dist_1 is not None:
                        aligned_target_dist_1[i][j][self.pad_id] = 1.0

        features.pop("per_step_logits")
        features.pop("per_step_indices")
        if "per_step_aligned_logits_0" in features:
            features.pop("per_step_aligned_logits_0")
            features.pop("per_step_aligned_indices_0")
        if "per_step_aligned_logits_1" in features:
            features.pop("per_step_aligned_logits_1")
            features.pop("per_step_aligned_indices_1")

        if self.training_args.distill_with_ref_model is True:
            features["base_target_dist"] = base_target_dist
        else:
            features.pop("metric_ce")
        if aligned_target_dist_0 is not None:
            features["aligned_target_dist_0"] = aligned_target_dist_0
        elif "metric_ce_aligned_0" in features:
            features.pop("metric_ce_aligned_0")
        if aligned_target_dist_1 is not None:
            features["aligned_target_dist_1"] = aligned_target_dist_1
        elif "metric_ce_aligned_1" in features:
            features.pop("metric_ce_aligned_1")
        return features


"""Other functions."""

import gc
import logging
import os
import sys
from typing import Dict, List

import editdistance
import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)



IGNORE_TOKEN_ID = LabelSmoother.ignore_index
TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
    transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: "Ġ"
}


# get tokenizer
def get_tokenizer(model_name_or_path, cache_dir, model_max_length):
    kwargs = {
        "use_fast": False,
        "tokenizer_trust_remote_code": False,
        "model_trust_remote_code": False,
    }
    if "llama" in model_name_or_path.lower():
        kwargs["use_fast"] = False
        kwargs["tokenizer_trust_remote_code"] = False
        kwargs["model_trust_remote_code"] = False
    elif "mpt" in model_name_or_path.lower():
        kwargs["use_fast"] = True
        kwargs["tokenizer_trust_remote_code"] = True
        kwargs["model_trust_remote_code"] = True
    elif "pythia" in model_name_or_path.lower():
        kwargs["use_fast"] = True
        kwargs["tokenizer_trust_remote_code"] = True
        kwargs["model_trust_remote_code"] = True
    elif "gemma" in model_name_or_path.lower():
        kwargs["use_fast"] = True
        kwargs["tokenizer_trust_remote_code"] = True
        kwargs["model_trust_remote_code"] = True
    elif "gpt" in model_name_or_path.lower():
        kwargs["use_fast"] = True
        kwargs["tokenizer_trust_remote_code"] = True
        kwargs["model_trust_remote_code"] = True
    else:
        raise NotImplementedError
    logger.info("Loading tokenizer.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=kwargs["use_fast"],
        trust_remote_code=kwargs["tokenizer_trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError
    logger.info(
        f"bos_token: {tokenizer.bos_token}, {tokenizer.bos_token_id} "
        f"eos_token: {tokenizer.eos_token}, {tokenizer.eos_token_id} "
        f"unk_token: {tokenizer.unk_token}, {tokenizer.unk_token_id} "
        f"pad_token: {tokenizer.pad_token}, {tokenizer.pad_token_id} "
    )
    return tokenizer, kwargs


# get base or peft model
def get_base_model(args, trust_remote_code):
    logger.info("Loading base model.")
    if args.use_flash_attn and "mpt" in args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(
            args.model_name_or_path, trust_remote_code=trust_remote_code
        )
        config.attn_config["attn_impl"] = "triton"
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            trust_remote_code=trust_remote_code,
            config=config,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            trust_remote_code=trust_remote_code,
        )
    return model


def find_all_linear_names(args, model):
    import bitsandbytes as bnb

    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_last_checkpoint_for_lora(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        is_completed = os.path.exists(os.path.join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(
                os.path.join(checkpoint_dir, filename)
            ) and filename.startswith("checkpoint"):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = os.path.join(checkpoint_dir, f"checkpoint-{max_step}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def get_accelerate_model(args, checkpoint_dir, trust_remote_code):
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer
    from transformers import BitsAndBytesConfig

    n_gpus = torch.cuda.device_count()
    max_memory = f"{args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )
    logger.info("Loading base model.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        ),
        trust_remote_code=trust_remote_code,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        logger.info("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(
            model, os.path.join(checkpoint_dir, "adapter_model"), is_trainable=True
        )
    else:
        logger.info("Adding lora module.")
        modules = find_all_linear_names(args, model)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


# save base or peft model
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info("Saving PEFT checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def release_model_and_tensor(model):
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def dict_to_list(examples):
    return [
        {key: examples[key][i] for key in examples}
        for i in range(len(examples[next(iter(examples))]))
    ]


def list_to_dict(examples):
    return {key: [d[key] for d in examples] for key in examples[0].keys()}


class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"No such attribute: {key}")


def dtw(series_1, series_2, norm_func=np.linalg.norm):
    """Use dynamic time wrapping to align to tokenizers, modified from:
    https://github.com/talcs/simpledtw/blob/master/simpledtw.py"""
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix


def transform_step_logits(
    base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
    base_model_vocab: Dict[str, int],
    base_model_input_ids: List[int],
    blending_model_input_ids: List[int],
    blending_model_per_step_logits: List[List[float]],
    blending_model_per_step_indices: List[List[int]],
    vocab_align_type: str = "hard",
    blending_to_base_mapping: Dict[str, str] = None,
):
    """Align blending model per step logits & indices with base model."""
    base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
    blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
        blending_model_input_ids
    )
    base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        base_model_tokenizer.__class__
    ]
    blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
        blending_model_tokenizer.__class__
    ]

    def dist_fn(a, b):
        """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
        aa = a.replace(blending_model_special_token, "")
        bb = b.replace(base_model_special_token, "")
        dist = editdistance.eval(aa, bb)
        return dist

    _, _, _, base_to_blending, _ = dtw(
        blending_model_tokens, base_model_tokens, norm_func=dist_fn
    )
    aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = (
        [],
        [],
    )
    for i, blending_idx in enumerate(base_to_blending):
        aligned_blending_model_per_step_logit = []
        aligned_blending_model_per_step_index = []
        if len(blending_idx) == 1:  # one base token map to one blending token
            j = blending_idx[0]
            base_token = base_model_tokens[i]
            blending_token = blending_model_tokens[j].replace(
                blending_model_special_token, base_model_special_token
            )
            if (
                (
                    blending_model_tokenizer.__class__
                    == transformers.GPTNeoXTokenizerFast
                    or blending_model_tokenizer.__class__
                    == transformers.GPT2TokenizerFast
                )
                and i == 0
                and base_token.startswith(base_model_special_token)
                and not blending_token.startswith(base_model_special_token)
            ):
                blending_token = (
                    base_model_special_token + blending_token
                )  # special case for mpt
            if vocab_align_type == "hard":
                if (
                    base_token == blending_token
                ):  # find the aligned mapping, use the corresponding logits
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(
                        blending_model_per_step_logits[j],
                        blending_model_per_step_indices[j],
                    ):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                            [blending_index]
                        )[0].replace(
                            blending_model_special_token, base_model_special_token
                        )
                        if blending_t in base_model_vocab:
                            aligned_index = base_model_vocab[
                                blending_t
                            ]  # the index of the token in base model vocab
                            if (
                                aligned_index
                                not in aligned_blending_model_per_step_index
                            ):
                                aligned_blending_model_per_step_index.append(
                                    aligned_index
                                )
                                aligned_blending_model_per_step_logit.append(
                                    blending_logit
                                )
                else:  # find error aligned mapping, use the one-hot logits
                    aligned_blending_model_per_step_index.append(
                        base_model_vocab[base_token]
                    )
                    aligned_blending_model_per_step_logit.append(1.0)
            elif vocab_align_type == "soft":
                if (base_token == blending_token) or (
                    blending_token in blending_to_base_mapping
                    and base_token == blending_to_base_mapping[blending_token]
                ):  # find the aligned mapping, use the corresponding logits
                    # the logits and indices at this step
                    for blending_logit, blending_index in zip(
                        blending_model_per_step_logits[j],
                        blending_model_per_step_indices[j],
                    ):
                        # the token corresponds to the logit and indices
                        blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                            [blending_index]
                        )[0].replace(
                            blending_model_special_token, base_model_special_token
                        )
                        blending_t = blending_to_base_mapping[blending_t]
                        if blending_t in base_model_vocab:
                            aligned_index = base_model_vocab[
                                blending_t
                            ]  # the index of the token in base model vocab
                            if (
                                aligned_index
                                not in aligned_blending_model_per_step_index
                            ):
                                aligned_blending_model_per_step_index.append(
                                    aligned_index
                                )
                                aligned_blending_model_per_step_logit.append(
                                    blending_logit
                                )
                        else:
                            logger.warning(
                                f"blending_t: {blending_t} not in base_model_vocab!"
                            )
                else:  # find error aligned mapping, use the one-hot logits
                    aligned_blending_model_per_step_index.append(
                        base_model_vocab[base_token]
                    )
                    aligned_blending_model_per_step_logit.append(1.0)
            else:
                logger.warning(
                    f"The vocab_align_type: '{vocab_align_type}' is not support!"
                )
                raise NotImplementedError
        else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
            base_token = base_model_tokens[i]
            aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
            aligned_blending_model_per_step_logit.append(1.0)
        aligned_blending_model_per_step_indices.append(
            aligned_blending_model_per_step_index
        )
        aligned_blending_model_per_step_logits.append(
            aligned_blending_model_per_step_logit
        )
    return (
        aligned_blending_model_per_step_logits,
        aligned_blending_model_per_step_indices,
    )


########################################################






logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forward for each teacher model to get logits of each token."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The input data dir. Should contain the training files.",
    )
    parser.add_argument(
        "--dataset_save_dir",
        type=str,
        required=True,
        help="The local dir to save processed data.",
    )
    parser.add_argument(
        "--dataset_sample_prop",
        type=float,
        default=None,
        help="The prop to sample dataset. Debugging only.",
    )
    parser.add_argument(
        "--dataset_split_num",
        type=int,
        default=None,
        help="The number to split dataset.",
    )
    parser.add_argument(
        "--dataset_index", type=int, default=None, help="The index of current dataset."
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache dir.")
    parser.add_argument(
        "--model_max_length", type=int, default=2048, help="The model max length."
    )
    parser.add_argument(
        "--training_mode", type=str, default="full", help="full or qlora."
    )
    parser.add_argument(
        "--load_in_half", type=str, default="none", help="none or fp16 or bf16."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=80,
        help="The number of processes to do data loading.",
    )
    parser.add_argument(
        "--top_k_logits", type=int, default=10, help="The number of logit for saving."
    )
    parser.add_argument(
        "--save_per_token_metric", action="store_true", help="Save per token metric."
    )
    parser.add_argument("--no_assert", action="store_true", help="Delete the assert.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Data processing args: {args}")
    dataset_mapping = load_from_disk(args.dataset)
    if args.dataset_sample_prop is not None:
        logger.info(f"Sample prop: {args.dataset_sample_prop}.")
        for k, v in dataset_mapping.items():
            select_size = int(len(v) * args.dataset_sample_prop)
            select_dataset = v.select(range(select_size))
            dataset_mapping[k] = select_dataset
            logger.info(f"{k}: select_size: {len(select_dataset)}")
    if args.dataset_split_num is not None:
        args.dataset_split_num = int(args.dataset_split_num)
        args.dataset_index = int(args.dataset_index)
        logger.info(
            f"Split num: {args.dataset_split_num}; Split index: {args.dataset_index}."
        )
        for k, v in dataset_mapping.items():
            select_size = int(len(v) / args.dataset_split_num)
            start_index = args.dataset_index * select_size
            end_index = (
                (args.dataset_index + 1) * select_size
                if args.dataset_index + 1 < args.dataset_split_num
                else len(v)
            )

            select_dataset = v.select(range(start_index, end_index))
                
            dataset_mapping[k] = select_dataset
            logger.info(
                f"{k}: start_index: {start_index}, end_index: {end_index}, select_size: {len(select_dataset)}"
            )

    tokenizer, _ = get_tokenizer(
        args.model_name_or_path, args.cache_dir, args.model_max_length
    )

    def tokenize_dataset(examples):
        text: List[str] = examples["text"]
        text = [x + tokenizer.eos_token for x in text]  # add eos in the end
        tknz_text = tokenizer(
            text,
            add_special_tokens=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        tknz_text["labels"] = tknz_text["input_ids"].copy()
        return tknz_text

    tokenized_dataset = dataset_mapping.map(
        tokenize_dataset,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc="Tokenize the dataset.",
    )

    model_args = {
        "model_name_or_path": args.model_name_or_path,
        "cache_dir": args.cache_dir,
        "model_max_length": args.model_max_length,
        "training_mode": args.training_mode,
        "use_flash_attn": False,
    }

    _, model = load_tokenizer_and_model(AttrDict(model_args))
    if args.load_in_half == "fp16":
        model = model.half()
    elif args.load_in_half == "bf16":
        model = model.to(dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    collate_function = DataCollatorForSeq2Seq(
        tokenizer,
        padding="max_length",
        max_length=args.model_max_length,
        label_pad_token_id=IGNORE_TOKEN_ID,
    )

    def forward_for_logits(examples):
        features = dict_to_list(examples)
        features = collate_function(features)
        if model.device.type == "cuda":
            input_ids = features["input_ids"].cuda()
            attention_mask = features["attention_mask"].cuda()
            labels = features["labels"].cuda()
        else:
            input_ids = features["input_ids"]
            attention_mask = features["attention_mask"]
            labels = features["labels"]
        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.to(torch.float16)
            metric_ce = (
                F.cross_entropy(
                    logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                    labels[..., 1:].contiguous().view(-1),
                    reduction="none",
                )
                .view(logits.size(0), -1)
                .to(torch.float16)
            )
            if args.save_per_token_metric:
                examples["per_step_metric_ce"] = metric_ce.cpu()
            metric_ce = (metric_ce * attention_mask[..., 1:]).sum(
                dim=-1
            ) / attention_mask[..., 1:].sum(dim=-1)
            logits = logits.cpu()
            metric_ce = metric_ce.cpu()
            if not args.no_assert:
                assert not bool(torch.isnan(logits).any().item())
                assert not bool(torch.isnan(metric_ce).any().item())
            input_ids.cpu()
            del input_ids
            attention_mask.cpu()
            del attention_mask
            labels.cpu()
            del labels
        if args.top_k_logits:
            top_k_logits, top_k_indices = torch.topk(logits.cuda(), k=args.top_k_logits)
            top_k_logits = top_k_logits.cpu()
            top_k_indices = top_k_indices.cpu()
            examples["per_step_logits"] = top_k_logits
            examples["per_step_indices"] = top_k_indices
        else:
            logger.warning("Saving all logits is too large!")
            raise ValueError
        examples["metric_ce"] = metric_ce
        return examples

    logits_datasets = tokenized_dataset.map(
        forward_for_logits,
        batched=True,
        batch_size=args.batch_size,
        writer_batch_size=1000,
        num_proc=None,
        load_from_cache_file=True,
        desc="Forward and get logits of the dataset.",
    )
    release_model_and_tensor(model)
    logits_datasets.save_to_disk(args.dataset_save_dir)
