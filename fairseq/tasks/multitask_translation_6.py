# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import warnings
from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional, List
from argparse import Namespace
from omegaconf import II, DictConfig
from collections import OrderedDict

import numpy as np
from fairseq import metrics, utils, search, tokenizer
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    TokenBlockDataset,
    NestedDictionaryDataset,
    MonolingualDataset,
    RoundRobinZipDatasets,
    MaskTokensDataset,
    SortDataset,
    IdDataset,
    RightPadDataset,
    NumSamplesDataset,
    NumelDataset,
    iterators, 
    FairseqDataset,
    NoisingDataset,
    Dictionary
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.data.encoders.utils import get_whole_word_mask

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)

def _get_mlm_dataset_key(lang_pair):
    return "mlm:" + lang_pair

def _get_lm_dataset_key(lang_pair):
    return "lm:" + lang_pair

def _get_denoising_dataset_key(lang_pair):
    return "denoising:" + lang_pair

# ported from UnsupervisedMT
def parse_lambda_config(x):
    """
    Parse the configuration of lambda coefficient (for scheduling).
    x = "3"                  # lambda will be a constant equal to x
    x = "0:1,1000:0"         # lambda will start from 1 and linearly decrease
                             # to 0 during the first 1000 iterations
    x = "0:0,1000:0,2000:1"  # lambda will be equal to 0 for the first 1000
                             # iterations, then will linearly increase to 1 until iteration 2000
    """
    split = x.split(",")
    if len(split) == 1:
        return float(x), None
    else:
        split = [s.split(os.pathsep) for s in split]
        assert all(len(s) == 2 for s in split)
        assert all(k.isdigit() for k, _ in split)
        assert all(
            int(split[i][0]) < int(split[i + 1][0]) for i in range(len(split) - 1)
        )
        return float(split[0][1]), [(int(k), float(v)) for k, v in split]

def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "mt {} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )

def load_maskedlm_dataset(
    data_path,
    split,
    lang,
    lang_dict,
    combine,
    dataset_impl,
    upsample_primary,
    shorten_data_split_list,
    shorten_method,
    tokens_per_sample,
    sample_break_mode,
    mask_whole_words,
    mask_idx,
    mask_prob,
    leave_unmasked_prob,
    random_token_prob,
    freq_weighted_replacement,
    mask_multiple_length,
    mask_stdev,
    seed=0,
    tgt=None,
    epoch=1,
    monopath="mono"
):
    if split=='train':
        data_path=os.path.join(data_path, monopath)
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    datasets = []
    src = lang
    for k in itertools.count():
        # if split=='train':
        #     split_k = split + ".mono" + (str(k) if k > 0 else "")
        # else:
        # if split=='train' and k > 0:
        #     data_path=os.path.join(data_path_s, monopath)
        #     split_k = split + (str(k) if k-1 > 0 else "")
        # else:
        split_k = split + (str(k) if k > 0 else "")


        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        elif split_exists(split_k, src, "None", src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, "None"))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        dataset = data_utils.load_indexed_dataset(
            prefix + lang, lang_dict, dataset_impl
        )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            shorten_data_split_list,
            shorten_method,
            tokens_per_sample,
            seed,
        )
        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            tokens_per_sample - 1,  # one less for <s>
            pad=lang_dict.pad(),
            eos=lang_dict.eos(),
            break_mode=sample_break_mode,
        )
        logger.info("mlm loaded {} blocks from: {}".format(len(dataset), prefix + lang))        

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        # dataset = PrependTokenDataset(dataset, lang_dict.bos())

        datasets.append(dataset)

        # logger.info(
        #     "{} {} {}-{} {} examples".format(
        #         data_path, split_k, src, tgt, len(datasets[-1])
        #     )
        # )

        if not combine:
            break


    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        sample_ratios = [1] * len(datasets)
        sample_ratios[0] = upsample_primary
        dataset = ConcatDataset(datasets, sample_ratios)

    # create masked input and targets
    # mask_whole_words = (
    #     get_whole_word_mask(self.args, self.source_dictionary)
    #     if mask_whole_words
    #     else None
    # )

    src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
        dataset,
        lang_dict,
        pad_idx=lang_dict.pad(),
        mask_idx=mask_idx,
        seed=seed,
        mask_prob=mask_prob,
        leave_unmasked_prob=leave_unmasked_prob,
        random_token_prob=random_token_prob,
        freq_weighted_replacement=freq_weighted_replacement,
        mask_whole_words=mask_whole_words,
        mask_multiple_length=mask_multiple_length,
        mask_stdev=mask_stdev,
    )

    with data_utils.numpy_seed(seed + epoch):
        shuffle = np.random.permutation(len(dataset))

    return SortDataset(
        NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=lang_dict.pad(),
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": RightPadDataset(
                    tgt_dataset,
                    pad_idx=lang_dict.pad(),
                ),
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(src_dataset, reduce=True),
            },
            sizes=[src_dataset.sizes],
        ),
        sort_order=[
            shuffle,
            src_dataset.sizes,
        ],
    )

def load_lm_datasets(
    data_path,
    split,
    lang,
    lang_dict,
    combine,
    dataset_impl,
    upsample_primary,
    shorten_data_split_list,
    shorten_method,
    tokens_per_sample,
    sample_break_mode,
    targets,
    seed,
    add_bos_token,
    src=None,
    epoch=1,
    monopath="mono"
):
    if split=='train':
        data_path=os.path.join(data_path, monopath)
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    datasets = []
    tgt = lang
    for k in itertools.count():
        # else:
        # if split=='train' and k > 0:
        #     data_path=os.path.join(data_path_s, monopath)
        #     split_k = split + (str(k) if k-1 > 0 else "")
        # else:
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, tgt, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, tgt, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        elif split_exists(split_k, tgt, "None", tgt, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, "None"))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        dataset = data_utils.load_indexed_dataset(
            prefix + lang, lang_dict, dataset_impl
        )

        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, prefix + lang)
            )
        
        dataset = maybe_shorten_dataset(
            dataset,
            split,
            shorten_data_split_list,
            shorten_method,
            tokens_per_sample,
            seed,
        )

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            tokens_per_sample,
            pad=lang_dict.pad(),
            eos=lang_dict.eos(),
            break_mode=sample_break_mode,
            include_targets=True,
        )
        logger.info("lm loaded {} blocks from: {}".format(len(dataset), prefix + lang))     

        datasets.append(dataset)

        # logger.info(
        #     "{} {} {}-{} {} examples".format(
        #         data_path, split_k, src, tgt, len(datasets[-1])
        #     )
        # )

        if not combine:
            break

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        sample_ratios = [1] * len(datasets)
        sample_ratios[0] = upsample_primary
        dataset = ConcatDataset(datasets, sample_ratios)

    add_eos_for_other_targets = (
        sample_break_mode is not None
        and sample_break_mode != "none"
    )

    return MonolingualDataset(
        dataset=dataset,
        sizes=dataset.sizes,
        src_vocab=lang_dict,
        tgt_vocab=lang_dict,
        add_eos_for_other_targets=add_eos_for_other_targets,
        shuffle=True,
        targets=targets,
        add_bos_token=add_bos_token,
    )

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])

@dataclass
class MultiTaskTranslationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },
    )
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: str = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: str = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    only_eval_mt: bool = field(
        default=False, metadata={"help": "only valid mt in valid step"}
    )
    
    # new args for mlm
    mlm: bool = field(
        default=False, metadata={"help": "pad the source on the left"}
    )
    mlm_shorten_data_split_list: str = field(
        default="", metadata={"help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',}
    )

    mlm_shorten_method: str = field(
        default="none", 
        metadata={"help": "if not none, shorten sequences that exceed --tokens-per-sample",
                "choices": ["none", "truncate", "random_crop"]}
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"}
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_whole_words: bool = field(
        default=False, 
        metadata={"help":"mask whole words; you may also want to set --bpe"}
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"}
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"}
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"}
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"}
    )
    mlm_tokens_per_sample: int = field(
        default=512,
        metadata={"help": "max number of total tokens over all segments "
            "per sample for BERT dataset"}
    )
    mlm_sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="complete",
        metadata={ "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'}
    )

    # language modeling args
    lm_sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    lm_tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    lm_shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    lm_shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    # TODO common vars below add to parent
    seed: int = II("common.seed")
    # dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
    #     "dataset.dataset_impl"
    # )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")

    # new args for multi task training
    testing: bool = field(
        default=False, metadata={"help": "training mode or not"}
    )

    lambda_mt_config: str = field(
        default="1.0",
        metadata={
            "help": 'cross-entropy reconstruction coefficient (parallel data). '
                    'use fixed weight during training if set to floating point number. '
                    'use piecewise linear function over number of updates to schedule the '
                    'weight with the format: w0:step0,w1:step1,...'
        }
    )
    lambda_mlm_config: str = field(
        default="1.0",
        metadata={
            "help": 'cross-entropy reconstruction coefficient (parallel data). '
                    'use fixed weight during training if set to floating point number. '
                    'use piecewise linear function over number of updates to schedule the '
                    'weight with the format: w0:step0,w1:step1,...'
        }
    )
    lambda_lm_config: str = field(
        default="1.0",
        metadata={
            "help": 'cross-entropy reconstruction coefficient (parallel data). '
                    'use fixed weight during training if set to floating point number. '
                    'use piecewise linear function over number of updates to schedule the '
                    'weight with the format: w0:step0,w1:step1,...'
        }
    )
    lambda_srcdae_config: str = field(
        default="1.0",
        metadata={
            "help": 'cross-entropy reconstruction coefficient (parallel data). '
                    'use fixed weight during training if set to floating point number. '
                    'use piecewise linear function over number of updates to schedule the '
                    'weight with the format: w0:step0,w1:step1,...'
        }
    )
    lambda_tgtdae_config: str = field(
        default="1.0",
        metadata={
            "help": 'cross-entropy reconstruction coefficient (parallel data). '
                    'use fixed weight during training if set to floating point number. '
                    'use piecewise linear function over number of updates to schedule the '
                    'weight with the format: w0:step0,w1:step1,...'
        }
    )
    max_word_shuffle_distance: float = field(
        default=3.0,
        metadata={"help": "maximum word shuffle distance for denoising autoencoding data generation"}
    )
    word_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "word dropout probability for denoising autoencoding data generation"}
    )
    word_blanking_prob: float = field(
        default=0.2,
        metadata={"help": "word blanking probability for denoising autoencoding data generation"}
    )


    share_dictionary: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    monopath: str =field(
        default="mono",
        metadata={
            "help":"mono paths"
        }
    )





@register_task("multitasktranslation6", dataclass=MultiTaskTranslationConfig)
class MultitaskTranslationTask6(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: MultiTaskTranslationConfig

    def __init__(self, cfg: MultiTaskTranslationConfig, src_dict, tgt_dict, lm_targets=None):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.lambda_mt, self.lambda_mt_steps = parse_lambda_config(
            cfg.lambda_mt_config
        ) 
        self.lambda_mlm, self.lambda_mlm_steps = parse_lambda_config(
            cfg.lambda_mlm_config
        ) 
        self.lambda_lm, self.lambda_lm_steps = parse_lambda_config(
            cfg.lambda_lm_config
        ) 
        self.lambda_srcdae, self.lambda_srcdae_steps = parse_lambda_config(
            cfg.lambda_srcdae_config
        ) 
        self.lambda_tgtdae, self.lambda_tgtdae_steps = parse_lambda_config(
            cfg.lambda_tgtdae_config
        ) 
        
        if lm_targets is None:
            lm_targets = ["future"]
        self.lm_targets = lm_targets
        self.testing = cfg.testing
        self.only_eval_mt = cfg.only_eval_mt

        if cfg.share_dictionary:
            self.src_mask_idx = src_dict.add_symbol("<mask>")
            self.tgt_mask_idx = tgt_dict.add_symbol("<mask>")
            assert self.src_mask_idx == self.tgt_mask_idx
        else:
            self.src_mask_idx = src_dict.add_symbol("<mask>")
        

    @classmethod
    def setup_task(cls, cfg: MultiTaskTranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        # self.src_mask_idx = src_dict.add_symbol("<mask>")
        # self.tgt_mask_idx = tgt_dict.add_symbol("<mask>")
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        # upgrade old checkpoints
        if getattr(cls, "exclude_self_target", False):
            args.self_target = False

        lm_targets = []
        if getattr(cls, "self_target", False):
            lm_targets.append("self")
        if getattr(cls, "future_target", False):
            lm_targets.append("future")
        if getattr(cls, "past_target", False):
            lm_targets.append("past")
        if len(lm_targets) == 0:
            # standard language modeling
            lm_targets = ["future"]

        return cls(cfg, src_dict, tgt_dict, lm_targets)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        def load_indexed_dataset(path, dictionary):
            return data_utils.load_indexed_dataset(
                path, dictionary, self.cfg.dataset_impl
            )

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        mt_dataset = None
        if (
            self.lambda_mt > 0.0 or self.lambda_mt_steps is not None
        ) or split.startswith("valid") or split.startswith("test"):
            mt_dataset = load_langpair_dataset(
                data_path,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=(split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
            )

        mlm_dataset = None
        if (
            self.lambda_mlm > 0.0 or self.lambda_mlm_steps is not None
        ) and (split.startswith("train") or split.startswith("valid")):
            # create masked input and targets
            mask_whole_words = (
                get_whole_word_mask(self.cfg, self.src_dict)
                if self.cfg.mask_whole_words
                else None
            )
            mlm_dataset = load_maskedlm_dataset(
                data_path,
                split,
                src,
                self.src_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                shorten_data_split_list  = self.cfg.mlm_shorten_data_split_list,
                shorten_method = self.cfg.mlm_shorten_method,
                tokens_per_sample = self.cfg.mlm_tokens_per_sample,
                sample_break_mode = self.cfg.mlm_sample_break_mode,
                mask_whole_words = mask_whole_words,
                mask_idx = self.src_mask_idx,
                mask_prob = self.cfg.mask_prob,
                leave_unmasked_prob = self.cfg.leave_unmasked_prob,
                random_token_prob = self.cfg.random_token_prob,
                freq_weighted_replacement = self.cfg.freq_weighted_replacement,
                mask_multiple_length = self.cfg.mask_multiple_length,
                mask_stdev = self.cfg.mask_stdev,
                seed = self.cfg.seed,
                tgt=tgt,
                epoch=epoch,
                monopath=self.cfg.monopath,
            )
            
        lm_dataset = None
        if (
            self.lambda_lm > 0.0 or self.lambda_lm_steps is not None
        ) and (split.startswith("train") or split.startswith("valid")):

            lm_dataset = load_lm_datasets(
                data_path,
                split,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                shorten_data_split_list = self.cfg.lm_shorten_data_split_list,
                shorten_method = self.cfg.lm_shorten_method,
                tokens_per_sample = self.cfg.lm_tokens_per_sample,
                sample_break_mode = self.cfg.lm_sample_break_mode,
                targets = self.lm_targets,
                seed = self.cfg.seed,
                add_bos_token = self.cfg.add_bos_token,
                src=src,
                epoch=epoch,
                monopath=self.cfg.monopath,
            )
        
        src_dae_dataset = None
        # denoising autoencoder
        if (
            self.lambda_srcdae > 0.0 or self.lambda_srcdae_steps is not None
        ) and (split.startswith("train") or split.startswith("valid")):
            filename = os.path.join(
                data_path, self.cfg.monopath, "{}.{}-None.{}".format(split, src, src)
            )
            if split.startswith("valid"):
                filename = os.path.join(
                    data_path, "{}.{}-{}.{}".format(split, src, tgt, src)
                )
            src_dataset1 = load_indexed_dataset(filename, self.src_dict)
            src_dataset2 = load_indexed_dataset(filename, self.src_dict)
            noising_dataset = NoisingDataset(
                src_dataset1,
                self.src_dict,
                seed=1,
                max_word_shuffle_distance=self.cfg.max_word_shuffle_distance,
                word_dropout_prob=self.cfg.word_dropout_prob,
                word_blanking_prob=self.cfg.word_blanking_prob,
            )
      
            src_dae_dataset = LanguagePairDataset(
                noising_dataset,
                src_dataset1.sizes,
                self.src_dict,
                src_dataset2,
                src_dataset2.sizes,
                self.src_dict,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
            )
            logger.info(
                "dae-{}: {} {} {} examples".format(
                    src,
                    data_path,
                    split,
                    len(src_dae_dataset),
                )
            )

        tgt_dae_dataset = None
        # denoising autoencoder
        if (
            self.lambda_tgtdae > 0.0 or self.lambda_tgtdae_steps is not None
        ) and (split.startswith("train") or split.startswith("valid")):
            filename = os.path.join(
                data_path, self.cfg.monopath, "{}.{}-None.{}".format(split, tgt, tgt)
            )
            if split.startswith("valid"):
                filename = os.path.join(
                    data_path, "{}.{}-{}.{}".format(split, src, tgt, tgt)
                )
            tgt_dataset1 = load_indexed_dataset(filename, self.tgt_dict)
            tgt_dataset2 = load_indexed_dataset(filename, self.tgt_dict)
            noising_dataset = NoisingDataset(
                tgt_dataset1,
                self.tgt_dict,
                seed=1,
                max_word_shuffle_distance=self.cfg.max_word_shuffle_distance,
                word_dropout_prob=self.cfg.word_dropout_prob,
                word_blanking_prob=self.cfg.word_blanking_prob,
            )
      
            tgt_dae_dataset = LanguagePairDataset(
                noising_dataset,
                tgt_dataset1.sizes,
                self.tgt_dict,
                tgt_dataset2,
                tgt_dataset2.sizes,
                self.tgt_dict,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
            )
            logger.info(
                "dae-{}: {} {} {} examples".format(
                    tgt,
                    data_path,
                    split,
                    len(tgt_dae_dataset),
                )
            )

        
        datasetlist = []
        if mt_dataset is not None:
            datasetlist += [('mt', mt_dataset)]
        if lm_dataset is not None:
            datasetlist += [('lm', lm_dataset)]
        if mlm_dataset is not None:
            datasetlist += [('mlm', mlm_dataset)]

        if src_dae_dataset is not None:
            datasetlist += [('srcdae', src_dae_dataset)]
        if tgt_dae_dataset is not None:
            datasetlist += [('tgtdae', tgt_dae_dataset)]
        

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(datasetlist),
            eval_key=None
            if not self.testing
            else "mt",
        )
        assert len(datasetlist) > 0
        logger.info(
            "{} datasets ".format(
                len(datasetlist)
            )
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        if update_num > 0:
            self.update_step(update_num)
        from collections import defaultdict

        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)

        def forward_backward(model, samples, logging_output_key, weight, mode):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            loss, sample_size, logging_output = criterion(model, samples, mode=mode)
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{logging_output_key}:{k}"] += logging_output[k]

        if self.lambda_mt > 0.0:
            forward_backward(model, sample['mt'], 'mt', self.lambda_mt, 'mt')
        
        if self.lambda_srcdae > 0.0:
            for p in model.decoder.parameters():
                p.requires_grad = False
            for p in model.encoder.embed_tokens.parameters():
                p.requires_grad = True
            forward_backward(model, sample['srcdae'], 'srcdae', self.lambda_srcdae, 'srcdae')
            for p in model.decoder.parameters():
                p.requires_grad = True

        if self.lambda_tgtdae > 0.0:
            for p in model.encoder.parameters():
                p.requires_grad = False
            for p in model.encoder.embed_tokens.parameters():
                p.requires_grad = True
            forward_backward(model, sample['tgtdae'], 'tgtdae', self.lambda_tgtdae, 'tgtdae')
            for p in model.encoder.parameters():
                p.requires_grad = True

        if self.lambda_mlm > 0.0:
            forward_backward(model.encoder, sample['mlm'], 'mlm', self.lambda_mlm, 'mlm')

        if self.lambda_lm > 0.0:
            forward_backward(model.decoder, sample['lm'], 'lm', self.lambda_lm, 'lm')
        

        
        return agg_loss, agg_sample_size, agg_logging_output

    def update_step(self, num_updates):
        def lambda_step_func(config, n_iter):
            """
            Update a lambda value according to its schedule configuration.
            """
            ranges = [
                i
                for i in range(len(config) - 1)
                if config[i][0] <= n_iter < config[i + 1][0]
            ]
            if len(ranges) == 0:
                assert n_iter >= config[-1][0]
                return config[-1][1]
            assert len(ranges) == 1
            i = ranges[0]
            x_a, y_a = config[i]
            x_b, y_b = config[i + 1]
            return y_a + (n_iter - x_a) * float(y_b - y_a) / float(x_b - x_a)

        if self.lambda_mt_steps is not None:
            self.lambda_mt = lambda_step_func(
                self.lambda_mt_steps, num_updates
            )
        if self.lambda_mlm_steps is not None:
            self.lambda_mlm = lambda_step_func(
                self.lambda_mlm_steps, num_updates
            )
        if self.lambda_lm_steps is not None:
            self.lambda_lm = lambda_step_func(self.lambda_lm_steps, num_updates)

        if self.lambda_srcdae_steps is not None:
            self.lambda_srcdae = lambda_step_func(self.lambda_tgtdae_steps, num_updates)
        if self.lambda_tgtdae_steps is not None:
            self.lambda_tgtdae = lambda_step_func(self.lambda_tgtdae_steps, num_updates)

    def valid_step(self, sample, model, criterion):
        modes = ['mt', 'mlm', 'lm', 'srcdae', 'tgtdae'] if not self.only_eval_mt else ['mt']
        model.eval()
        with torch.no_grad():
            from collections import defaultdict

            agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
            for mode in modes:
                if(
                    mode not in sample
                    or sample[mode] is None
                    or len(sample[mode]) == 0
                ):
                    continue
                if mode == 'mt' or mode == 'srcdae' or mode == 'tgtdae':
                    valid_model = model
                if mode == 'mlm':
                    valid_model = model.encoder
                if mode == 'lm':
                    valid_model = model.decoder
                loss, sample_size, logging_output = criterion(valid_model, sample[mode], mode=mode)
                agg_loss += loss.data.item()
                agg_sample_size += sample_size
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                    agg_logging_output[f"{mode}:{k}"] += logging_output[k]
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample['mt'], model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return agg_loss, agg_sample_size, agg_logging_output
                

    # def valid_step(self, sample, model, criterion):
    #     loss, sample_size, logging_output = super().valid_step(sample['mt'], model, criterion)
    #     if self.cfg.eval_bleu:
    #         bleu = self._inference_with_bleu(self.sequence_generator, sample['mt'], model)
    #         logging_output["_bleu_sys_len"] = bleu.sys_len
    #         logging_output["_bleu_ref_len"] = bleu.ref_len
    #         # we split counts into separate entries so that they can be
    #         # summed efficiently across workers using fast-stat-sync
    #         assert len(bleu.counts) == EVAL_BLEU_ORDER
    #         for i in range(EVAL_BLEU_ORDER):
    #             logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
    #             logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
    #     return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)



    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])


    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            # if not getattr(args, "crossatt", False) and not getattr(args, "num_lm_layers", False) and not getattr(args, "cross_self_attention", False):
            #     from fairseq.momer_sequence_generator import MomerSequenceGenerator
            #     seq_gen_cls = MomerSequenceGenerator
            # el
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )
