# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq.models.transformer import (
    TransformerModel,
    TransformerDecoder,
    TransformerEncoder,
    base_architecture,
    register_model,
    transformer_iwslt_de_en,
)
from fairseq.modules import LayerNorm
from fairseq.models import register_model_architecture
from fairseq import utils
from omegaconf import DictConfig
from torch import Tensor
import torch.nn.functional as F


logger = logging.getLogger(__name__)



class LMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class MLMTransformerEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.mlm = getattr(args, "mlm", False)
        self.encoder_out_layer = getattr(args, "encoder_out_layer", -1)
        if self.encoder_out_layer == -1:
            self.encoder_out_layer = args.encoder_layers
            self.mt_layer_norm = None
        else:
            assert self.encoder_out_layer <= args.encoder_layers and self.encoder_out_layer > 0
            if self.encoder_out_layer < args.encoder_layers and args.encoder_normalize_before:
                self.mt_layer_norm = LayerNorm(embed_tokens.embedding_dim)
            else:
                self.mt_layer_norm = None

        if self.mlm:
            self.lm_head = LMHead(embed_tokens.embedding_dim, 
                len(dictionary),
                activation_fn=args.lm_head_activation_fn, # "relu", # add args
                weight=(
                    self.embed_tokens.weight
                ),
            )
    
    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)


    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        masked_tokens=None,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = []
        mt_x = None
        num_layers = len(self.layers)
        # encoder layers
        for idx, layer in enumerate(self.layers):
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            if idx + 1 == self.encoder_out_layer and self.encoder_out_layer < num_layers:
                mt_x = x

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        if mt_x is not None and self.mt_layer_norm is not None:
            mt_x = self.mt_layer_norm(mt_x)
        
        mlm_out = None
        if self.mlm and masked_tokens is not None:
            features = x.transpose(0, 1)
            mlm_out = self.output_layer(features, masked_tokens=masked_tokens)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `foward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x if mt_x is None else mt_x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "mlm_out": mlm_out,
        }

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]
