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


@register_model("mlm_lm_layerwise_selfatt_transformer_model")
class MLMLMLyaerWiseTransformerModel(TransformerModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if args.freeze_params is not None:
            self.freeze_params(args)

    def freeze_params(self, args):
        freeze_pattern = re.compile(args.freeze_params)
        for name, parameter in self.named_parameters():
            if freeze_pattern.search(name):
                parameter.requires_grad = False
                logger.info(f"Freeze: {name}")
        for name, parameter in self.named_parameters():
            if not freeze_pattern.search(name):
                logger.info(f"Unfreeze: {name}")

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument('--num-lm-layers', type=int, metavar='D', default=6,
                            help='The bottom N layers of decoder are language model.')
        parser.add_argument("--crossatt", action="store_true", default=False,
                            help="whether use cross attention layer for top (decoder layers - N) layers for encoder decoder interacting.")
        parser.add_argument('--freeze-params', type=str, metavar='D', default=None,
                            help='regular expression of parameters that need to be frozen')
        parser.add_argument('--transfer-params', type=str, metavar='D', default=None,
                            help='transfer params from pretrained models')
        parser.add_argument("--lm-fusion", action="store_true", default=False,
                            help="whether to fusion lm")
        parser.add_argument("--mlm", action="store_true", default=False,
                            help="whether to fusion lm")
        parser.add_argument('--lm-head-activation-fn', type=str, metavar='D', default='relu',
                            help='regular expression of parameters that need to be frozen')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return MLMTransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return BridgeTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def load_state_dict(
        self,
        state_dict,
        strict=False,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        if self.args.transfer_params is not None and "inference" not in vars(model_cfg):
            pretrained_model_prefix = [*state_dict][0].split('.')[0]
            pairs = self.args.transfer_params.split(',')
            for pair in pairs:
                from_param, to_param = pair.split(':')
                if from_param in state_dict:
                    # state_dict[to_param] = state_dict.pop(from_param)
                    state_dict[to_param] = state_dict[from_param]
                    logger.info(f"Transfer {from_param} to {to_param} in model [{pretrained_model_prefix}]")
        return torch.nn.Module.load_state_dict(self, state_dict, strict=False)

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

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
        self.mlm = args.mlm
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

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        mlm_out = None
        if self.mlm and masked_tokens is not None:
            features = x.transpose(0, 1)
            mlm_out = self.output_layer(features, masked_tokens=masked_tokens)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `foward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
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

class BridgeTransformerDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        self.num_lm_layers = args.num_lm_layers
        self.crossatt = args.crossatt
        no_encoder_attn_layers = [str(i) for i in range(self.num_lm_layers)] \
            if args.num_lm_layers > 0 else []
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn=True)
                if str(layer) in no_encoder_attn_layers else self.build_decoder_layer(args, no_encoder_attn=not self.crossatt)
                for layer in range(args.decoder_layers)
            ]
        )
        for i in range(self.num_lm_layers):
            self.layers[i].cross_self_attention = False
        self.lm_layer_norm = None
        self.lm_output_projection = None
        if args.lm_fusion:
            self.lm_layer_norm = LayerNorm(self.embed_tokens.embedding_dim)
            self.lm_output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.lm_output_projection.weight = self.embed_tokens.weight

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        lm_incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        mt_incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        forward_lm: bool = False,
        forward_mt: bool = False,
        only_return_lm_output: bool = False,
        cat_encoder_out: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            lm_incremental_state = lm_incremental_state,
            mt_incremental_state = mt_incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            only_return_lm_output=only_return_lm_output,
            cat_encoder_out=cat_encoder_out
        )
        if only_return_lm_output:
            x = self.lm_output_projection(x)
            return x, extra

        if not features_only:
            x = self.output_layer(x)
            if self.lm_output_projection:
                lm_state = self.lm_output_projection(extra['lm_state'])
                extra['lm_out'] = lm_state
                x = x + lm_state
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        lm_incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        mt_incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        forward_lm: bool = False,
        forward_mt: bool = False,
        only_return_lm_output: bool = False,
        cat_encoder_out: bool = False,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            lm_incremental_state,
            mt_incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            only_return_lm_output,
            cat_encoder_out
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        lm_incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        mt_incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        forward_lm: bool = False,
        forward_mt: bool = False,
        only_return_lm_output: bool = False,
        cat_encoder_out: bool = False,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x_dim = x.size(0)
        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
    
        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        lm_state = None
        for idx, layer in enumerate(self.layers):
            if idx == self.num_lm_layers and mt_incremental_state is not None and cat_encoder_out and not self.crossatt:
                x = torch.concat([encoder_out["encoder_out"][0], x], dim=0)
                self_attn_padding_mask = torch.concat([encoder_out['encoder_padding_mask'][0], 
                                                prev_output_tokens.eq(self.padding_idx)], dim=1)   
            if incremental_state is None and not full_context_alignment:
                if idx == self.num_lm_layers and not self.crossatt:
                    # x = torch.concat([encoder_out["encoder_out"][0], x], dim=0)
                    # self_attn_padding_mask = torch.concat([encoder_out['encoder_padding_mask'][0], prev_output_tokens.eq(self.padding_idx)], dim=1)
                    self_attn_mask = self.buffered_future_mask_with_prev_tensor(
                        encoder_out["encoder_out"][0], x_dim)
                elif idx > self.num_lm_layers and not self.crossatt:
                    self_attn_mask = self.buffered_future_mask_with_prev_tensor(
                        encoder_out["encoder_out"][0], x_dim)
                else:
                    self_attn_mask = self.buffered_future_mask(x)
            # if (incremental_state is None or mt_incremental_state is None) and not full_context_alignment:
            #     self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0 and idx >= self.num_lm_layers)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            #            incremental_state if idx < self.num_lm_layers else mt_incremental_state,
            inner_states.append(x[-x_dim:])
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

            if self.layers[idx].encoder_attn is None and \
                idx+1 < len(self.layers) and idx + 1 == self.num_lm_layers:
                lm_state = x
                if only_return_lm_output is True:
                    break
                    
        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)
       
        if self.lm_layer_norm is not None:
            lm_state = self.lm_layer_norm(lm_state)
            lm_state = lm_state.transpose(0, 1)
            if only_return_lm_output is True:
                return lm_state, {"attn": attn, "inner_states": inner_states}
        x = x[-x_dim:]
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
 
        return x, {"attn": [attn], "inner_states": inner_states, "lm_state": lm_state}

    def forward_lm_layers(
        self,
        src_tokens,
        **kwargs,
    ):
        return self.forward(src_tokens, **kwargs)

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
    
    def buffered_future_mask_with_prev_tensor(self, prev_tensor, tgt_dim):
        prev_dim = prev_tensor.size(0)
        dim = prev_dim + tgt_dim
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == prev_tensor.device)
            or self._future_mask.size(0) < dim
        ):
            right_down_mat = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([tgt_dim, tgt_dim])), 1
            )
            right_top_mat = utils.fill_with_neg_inf(torch.zeros([prev_dim, tgt_dim]))
            left_top_mat = torch.zeros([prev_dim,prev_dim])
            left_down_mat = torch.zeros([tgt_dim,prev_dim])
            self._future_mask = torch.concat([torch.concat([left_top_mat, left_down_mat], dim=0),
                                              torch.concat([right_top_mat,right_down_mat], dim=0)], dim=1)
        self._future_mask = self._future_mask.to(prev_tensor)
        return self._future_mask[:dim,:dim]





@register_model_architecture("mlm_lm_layerwise_selfatt_transformer_model", "mlm_lm_layerwise_selfatt_transformer_model")
def base_mlm_lm_transformer_model(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.num_lm_layers = getattr(args, "num_lm_layers", 6)
    base_architecture(args)

@register_model_architecture("mlm_lm_layerwise_selfatt_transformer_model", "mlm_lm_layerwise_selfatt_transformer_model_iswlt")
def mlm_lm_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.num_lm_layers = getattr(args, "num_lm_layers", 6)
    transformer_iwslt_de_en(args)


@register_model_architecture("mlm_lm_layerwise_selfatt_transformer_model", "mlm_lm_layerwise_selfatt_transformer_model_mbert_mgpt")
def mlm_lm_transformer_mbert_mgpt(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.num_lm_layers = getattr(args, "num_lm_layers", 6)
    base_architecture(args)
