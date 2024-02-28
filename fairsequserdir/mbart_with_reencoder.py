from typing import Optional, Dict, List, NamedTuple

import logging
import re
import torch
import torch.nn as nn
from torch import Tensor
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq import checkpoint_utils
from collections import OrderedDict

from fairseq.models.bart import BARTModel, bart_large_architecture
from fairseq.models.transformer import transformer_vaswani_wmt_en_fr_big
from .multilingual_reencoder import ReEncoder


logger = logging.getLogger(__name__)


@register_model("mbart_with_reencoder")
class BARTModelwithReencoder(BARTModel):

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        
        self.transeferring=self.build_reencoder(args)

        if args.freeze_params is not None:
            self.freeze_params(args.freeze_params)
        
        
            
    @staticmethod
    def add_args(parser):
        BARTModel.add_args(parser)
        # parser.add_argument(
        #     "--pooler-dropout",
        #     type=float,
        #     metavar="D",
        #     help="dropout probability in the masked_lm pooler layers",
        # )
        # parser.add_argument(
        #     "--pooler-activation-fn",
        #     choices=utils.get_available_activation_fns(),
        #     help="activation function to use for pooler layer",
        # )
        # parser.add_argument(
        #     "--spectral-norm-classification-head",
        #     action="store_true",
        #     help="Apply spectral normalization on the classification head",
        # )
        # parser.add_argument(
        #     "--freeze-orig-params",
        #     action="store_true",
        #     help="Apply spectral normalization on the classification head",
        # )
        parser.add_argument(
            "--reencoder-layers",
            type=int,
            metavar="D",
            default=6,
            help="the number of reencoder layers"
        )
        parser.add_argument(
            "--freeze-params",
            type=str,
            metavar='D',
            default=None,
            help="Apply spectral normalization on the classification head",
        )
        parser.add_argument('--transfer-params', type=str, metavar='D', default=None,
                            help='transfer params from pretrained models')

        parser.add_argument('--reencoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--reencoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--reencoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--reencoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')

        parser.add_argument('--reencoder-layerdrop', type=float, metavar='D', default=0, help='LayerDrop probability for reencoder')

        parser.add_argument('--reencoder-layers-to-keep', default=None, help='which layers to *keep* when pruning as a comma-separated list')

        parser.add_argument(
            "--load-pretrained-encoder-from",
            default=None,
            type=str,
            metavar="PRETRAINED",
            help="path to pretrained mlm checkpoint",
        )

        parser.add_argument(
            "--load-pretrained-decoder-from",
            default=None,
            type=str,
            metavar="PRETRAINED",
            help="path to pretrained mlm checkpoint",
        )

        parser.add_argument(
            "--load-pretrained-reencoder-from",
            default=None,
            type=str,
            metavar="PRETRAINED",
            help="path to pretrained mlm checkpoint",
        )


    def freeze_params(self, freeze_params):
        freeze_pattern = re.compile(freeze_params)
        for name, parameter in self.named_parameters():
            if freeze_pattern.search(name):
                parameter.requires_grad = False
                logger.info(f"Freeze: {name}")
        for name, parameter in self.named_parameters():
            if not freeze_pattern.search(name):
                logger.info(f"Unfreeze: {name}")

    def load_state_dict(
        self,
        state_dict,
        strict=False,
        model_cfg = None,
        args = None,
    ):
        if self.args.transfer_params is not None and "inference" not in vars(model_cfg):
            pretrained_model_prefix = [*state_dict][0].split('.')[0]
            pairs = self.args.transfer_params.split(',')
            for pair in pairs:
                from_param, to_param = pair.split(':')
                if from_param in state_dict:
                    state_dict[to_param] = state_dict.pop(from_param)
                    # state_dict[to_param] = state_dict[from_param]
                    logger.info(f"Transfer {from_param} to {to_param} in model [{pretrained_model_prefix}]")
        missing_keys, unexpected_keys = super().load_state_dict(state_dict, False, model_cfg, args)
        # missing_keys, unexpected_keys = torch.nn.Module.load_state_dict(self, state_dict, strict)
        logger.info(f"pretrained model missing kyes | : {missing_keys}")
        logger.info(f"pretrained model unexpect kyes | : {unexpected_keys}")
        return missing_keys, unexpected_keys
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(f"load pretrained model for encoder")
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder =  TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

        if getattr(args, "load_pretrained_decoder_from", None):
            logger.info(f"load pretrained model for encoder")
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )

        return decoder
    
    @classmethod
    def build_reencoder(cls, args):
        reencoder = ReEncoder(args)

        if getattr(args, "load_pretrained_reencoder_from", None):
            logger.info(f"load pretrained model for reencoder")
            state = checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_reencoder_from)

            component_type = 'reencoder'
            component_state_dict = OrderedDict()
            for key in state["model"].keys():
                if key.startswith(component_type):
                    # encoder.input_layers.0.0.weight --> input_layers.0.0.weight
                    component_subkey = key[len(component_type) + 1 :]
                    component_state_dict[component_subkey] = state["model"][key]
            missing_keys,unexpect_keys=reencoder.load_state_dict(component_state_dict, strict=True)

            logger.info(f"pretrained reencoder missing kyes | : {missing_keys}")
            logger.info(f"pretrained reencoder unexpect kyes | : {unexpect_keys}")
        else:
            logger.info(f"not existing pretrained model for encoder...")
        return reencoder


    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )

        reencoder_out = self.transeferring(encoder_out["encoder_out"][0], encoder_out["encoder_padding_mask"][0])

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=reencoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        return decoder_out + (reencoder_out,)

    def forward_encoder_and_reencoder(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = True,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
        )
        reencoder_out = self.transeferring(encoder_out["encoder_out"][0], encoder_out["encoder_padding_mask"][0])
        return reencoder_out

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        reencoder_out = self.transeferring(encoder_out["encoder_out"][0], encoder_out["encoder_padding_mask"][0])
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=reencoder_out, **kwargs
        )
        return features
    

    def forward_encoder_and_reencoder_torchscript(self, net_input: Dict[str, Tensor]):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        if torch.jit.is_scripting():
            return self.forward_encoder_and_reencoder(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
            )
        else:
            return self.forward_encoder_and_reencoder_non_torchscript(net_input)

    @torch.jit.unused
    def forward_encoder_and_reencoder_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward_encoder_and_reencoder(**encoder_input)
    
    


@register_model_architecture("mbart_with_reencoder", "mbart_with_reencoder_large_for_ft")
def mbart_for_ft_large_architecture(args):
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

    args.reencoder_layers = getattr(args, "reencoder_layers", 6)
    args.reencoder_embed_dim = getattr(args, "reencoder_embed_dim", 1024)
    args.reencoder_ffn_embed_dim = getattr(args, "reencoder_ffn_embed_dim", 4 * 1024)
    args.reencoder_attention_heads = getattr(args, "reencoder_attention_heads", 16)
    args.reencoder_normalize_before = getattr(args, "reencoder_normalize_before", False)
    args.reencoder_layerdrop = getattr(args, "reencoder_layerdrop", 0)
    args.reencoder_layers_to_keep = getattr(args, "reencoder_layers_to_keep", None)
    bart_large_architecture(args)

@register_model_architecture("mbart_with_reencoder", "mbart_with_reencoder_transformer_large_for_ft")
def mbart_for_ft_transfomer_large_architecture(args):
    args.reencoder_layers = getattr(args, "reencoder_layers", 6)
    args.reencoder_embed_dim = getattr(args, "reencoder_embed_dim", 1024)
    args.reencoder_ffn_embed_dim = getattr(args, "reencoder_ffn_embed_dim", 4 * 1024)
    args.reencoder_attention_heads = getattr(args, "reencoder_attention_heads", 16)
    args.reencoder_normalize_before = getattr(args, "reencoder_normalize_before", False)
    args.reencoder_layerdrop = getattr(args, "reencoder_layerdrop", 0)
    args.reencoder_layers_to_keep = getattr(args, "reencoder_layers_to_keep", None)
    transformer_vaswani_wmt_en_fr_big(args)
