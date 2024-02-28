# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import copy
import logging
import os
from typing import Any, Dict, Iterator, List
from omegaconf import open_dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import encoders


class MoNMTHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        self.models = [model]

        self.bpe = encoders.build_bpe(cfg.bpe)

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(
        self, sentence: str, *addl_sentences, special_token=None, no_separator=False
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        # bpe_sentence = "<s> " + self.bpe.encode(sentence) + " </s>"
        # for s in addl_sentences:
        #     bpe_sentence += " </s>" if not no_separator else ""
        #     bpe_sentence += " " + self.bpe.encode(s) + " </s>"

        if special_token != None:
            bpe_sentence = self.bpe.encode(sentence) + special_token
        else:
            bpe_sentence = self.bpe.encode(sentence)
        # for s in addl_sentences:
        #     bpe_sentence += "" if not no_separator else ""
        #     bpe_sentence += " " + self.bpe.encode(s) + " </s>"
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence, append_eos=False, add_if_not_exist=False
        )
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.bpe.decode(self.task.source_dictionary.string(s)) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def decode_with_bpe(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [
            self.task.source_dictionary.string(s) for s in sentences
        ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def extract_encoder_features2(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        # if tokens.size(-1) > self.model.max_positions():
        #     raise ValueError(
        #         "tokens exceeds maximum length: {} > {}".format(
        #             tokens.size(-1), self.model.max_positions()
        #         )
        #     )
        lengths = [tokens.size(-1)]
        features = self.model.encoder(
            tokens.to(device=self.device),
            lengths,
            return_all_hiddens=return_all_hiddens,
        )["encoder_out"][0]
        # if return_all_hiddens:
        #     # convert from T x B x C -> B x T x C
        #     inner_states = extra["inner_states"]
        #     return [inner_state.transpose(0, 1) for inner_state in inner_states]
        # else:
        return features  # just the last layer's features

    def extract_encoder_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        # if tokens.size(-1) > self.model.max_positions():
        #     raise ValueError(
        #         "tokens exceeds maximum length: {} > {}".format(
        #             tokens.size(-1), self.model.max_positions()
        #         )
        #     )
        lengths = [tokens.size(-1)]
        features = self.model.forward_encoder(
            tokens.to(device=self.device),
            lengths,
            return_all_hiddens=return_all_hiddens,
        )["encoder_out"][0]
        # if return_all_hiddens:
        #     # convert from T x B x C -> B x T x C
        #     inner_states = extra["inner_states"]
        #     return [inner_state.transpose(0, 1) for inner_state in inner_states]
        # else:
        return features  # just the last layer's features

    def extract_reencoder_features(
        self, tokens: torch.LongTensor, return_all_hiddens: bool = False
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        # if tokens.size(-1) > self.model.max_positions():
        #     raise ValueError(
        #         "tokens exceeds maximum length: {} > {}".format(
        #             tokens.size(-1), self.model.max_positions()
        #         )
        #     )
        lengths = [tokens.size(-1)]
        features = self.model.forward_encoder_and_reencoder(
            tokens.to(device=self.device),
            lengths,
            return_all_hiddens=return_all_hiddens,
        )["encoder_out"][0]
        # if return_all_hiddens:
        #     # convert from T x B x C -> B x T x C
        #     inner_states = extra["inner_states"]
        #     return [inner_state.transpose(0, 1) for inner_state in inner_states]
        # else:
        return features  # just the last layer's features

    def translate(
        self, sentences: List[str], beam: int = 5, verbose: bool = False, **kwargs
    ) -> List[str]:
        return self.sample(sentences, beam, verbose, **kwargs)

    def sample(
        self, sentences: List[str], beam: int = 1, verbose: bool = False, **kwargs
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], beam=beam, verbose=verbose, **kwargs)[0]
        tokenized_sentences = [self.encode(sentence) for sentence in sentences]
        batched_hypos = self.generate(tokenized_sentences, beam, verbose, **kwargs)
        return [self.decode(hypos[0]["tokens"]) for hypos in batched_hypos]

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(self.models, gen_args)

        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo["tokens"])
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    logger.info(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    if hypo["alignment"] is not None and getarg(
                        "print_alignment", False
                    ):
                        logger.info(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )
        return outputs

    def _build_batches(
        self, tokens: List[List[int]], skip_invalid_size_inputs: bool
    ) -> Iterator[Dict[str, Any]]:
        lengths = torch.LongTensor([t.numel() for t in tokens])
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=1024,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator