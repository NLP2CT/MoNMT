import logging
import os
import warnings
from argparse import Namespace
from typing import List
from dataclasses import dataclass, field

import torch
from fairseq import metrics, search, tokenizer, utils
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import gen_parser_from_dataclass
from omegaconf import DictConfig

from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from .generator_with_reencoder import SequenceGeneratorWithReencoder


logger = logging.getLogger(__name__)

@dataclass
class myTranslationConfig(TranslationConfig):
    add_mask_symbol: bool = field(
        default=False, metadata={"help": "add mask tok <mask> to dict"}
    )
    add_lang_symbol: bool = field(
        default=False, metadata={"help": "add lang tok e.g.[en] to dict"}
    )
    lang_tok_first_add: bool = field(
        default=False, metadata={"help": "add lang tok e.g.[en] to dict"}
    )
    share_dictionary: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    append_source_id: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    langs: str = field(
        default="", metadata={"help": "languages in this model, e.g. ar,de,en"},
    )



@register_task("my_translation_task", dataclass=myTranslationConfig)
class myTranslationTask(TranslationTask):

    def __init__(self, cfg: myTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.langs = None
        if cfg.langs != "":
            self.langs=cfg.langs
        self.share_dictionary = cfg.share_dictionary
        self.src_mask_idx=None
        self.tgt_mask_idx=None
        if self.langs is not None:
            langs = self.langs.strip().split(',')
            assert cfg.source_lang in langs and cfg.target_lang in langs
            if cfg.add_lang_symbol and cfg.add_mask_symbol:
                if cfg.lang_tok_first_add:
                    for l in langs:
                        tmp = src_dict.add_symbol("[{}]".format(l))
                        tmp = tgt_dict.add_symbol("[{}]".format(l))
                    tmp = src_dict.add_symbol("<mask>")
                    tmp = tgt_dict.add_symbol("<mask>")
                else:
                    tmp = src_dict.add_symbol("<mask>")
                    tmp = tgt_dict.add_symbol("<mask>")
                    for l in langs:
                        tmp = src_dict.add_symbol("[{}]".format(l))
                        tmp = tgt_dict.add_symbol("[{}]".format(l))
            elif cfg.add_lang_symbol:
                for l in langs:
                    tmp = src_dict.add_symbol("[{}]".format(l))
                    tmp = tgt_dict.add_symbol("[{}]".format(l))
            else:
                logger.info("langs is not none, but not add lang symbol from it")
        else:
            logger.info("langs is none")
        self.src_lg_id=None
        self.tgt_lg_id=None
        if cfg.add_lang_symbol and cfg.add_mask_symbol:
            if cfg.lang_tok_first_add:
                self.src_lg_id = src_dict.add_symbol("[{}]".format(cfg.source_lang))
                if self.share_dictionary:
                    tmp = src_dict.add_symbol("[{}]".format(cfg.target_lang))
                    tmp = tgt_dict.add_symbol("[{}]".format(cfg.source_lang))
                self.tgt_lg_id = tgt_dict.add_symbol("[{}]".format(cfg.target_lang))
                self.src_mask_idx = src_dict.add_symbol("<mask>")
                self.tgt_mask_idx = tgt_dict.add_symbol("<mask>")
            else:
                self.src_mask_idx = src_dict.add_symbol("<mask>")
                self.tgt_mask_idx = tgt_dict.add_symbol("<mask>")
                self.src_lg_id = src_dict.add_symbol("[{}]".format(cfg.source_lang))
                if self.share_dictionary:
                    tmp = src_dict.add_symbol("[{}]".format(cfg.target_lang))
                    tmp = tgt_dict.add_symbol("[{}]".format(cfg.source_lang))
                self.tgt_lg_id = tgt_dict.add_symbol("[{}]".format(cfg.target_lang))

        elif cfg.add_lang_symbol:
                self.src_lg_id = src_dict.add_symbol("[{}]".format(cfg.source_lang))
                if self.share_dictionary:
                    tmp = src_dict.add_symbol("[{}]".format(cfg.target_lang))
                    tmp = tgt_dict.add_symbol("[{}]".format(cfg.source_lang))
                self.tgt_lg_id = tgt_dict.add_symbol("[{}]".format(cfg.target_lang))

        elif cfg.add_mask_symbol:
            self.src_mask_idx = src_dict.add_symbol("<mask>")
            self.tgt_mask_idx = tgt_dict.add_symbol("<mask>")
        else:
            logger.info("be informed that no extra lang toks or mask toks")
        logger.info("the src id is {}, the tgt id is {}".format(self.src_lg_id, self.tgt_lg_id))

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.args=cfg


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
            eos=self.tgt_dict.index("[{}]".format(self.args.target_lang)) if getattr(args, "append_source_id", False) else None,
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )
