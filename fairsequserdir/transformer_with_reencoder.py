import torch
import argparse
import logging
import re
from typing import Optional, Dict, List, NamedTuple
import torch.nn as nn
import fairseq.checkpoint_utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerDecoder, TransformerEncoder, TransformerModel, base_architecture, transformer_vaswani_wmt_en_fr_big
from fairseq.models.roberta import model as roberta
from fairseq import checkpoint_utils
from collections import OrderedDict
from .multilingual_reencoder import ReEncoder
from torch import Tensor
from fairseq.models.momer import MLMTransformerEncoder, CLMTransformerDecoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)


@register_model("reencoder_transformer")
class ReEncoderTransformer(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        
        parser.add_argument('--freeze-params', type=str, metavar='D', default=None,
                            help='regular expression of parameters that need to be frozen')
        parser.add_argument('--transfer-params', type=str, metavar='D', default=None,
                            help='transfer params from pretrained models')

        parser.add_argument('--reencoder-embed-dim', type=int, metavar='N', help='encoder embedding dimension')
        parser.add_argument('--reencoder-ffn-embed-dim', type=int, metavar='N', help='encoder embedding dimension for FFN')
        parser.add_argument('--reencoder-attention-heads', type=int, metavar='N', help='num encoder attention heads')
        parser.add_argument('--reencoder-normalize-before', action='store_true', help='apply layernorm before each encoder block')

        parser.add_argument('--reencoder-layerdrop', type=float, metavar='D', default=0, help='LayerDrop probability for reencoder')

        parser.add_argument('--reencoder-layers-to-keep', default=None, help='which layers to *keep* when pruning as a comma-separated list')

        parser.add_argument(
            "--reencoder-layers",
            type=int,
            metavar="D",
            default=6,
            help="the number of reencoder layers"
        )
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
        

        # arguments for mlm and clm
        parser.add_argument("--mlm", action="store_true", default=False,
                            help="whether train mlm task")
        parser.add_argument("--lm-head-activation-fn", type=str, metavar='D', default='relu',
                            help="mlm head activate")
        parser.add_argument("--encoder-out-layer", type=int, metavar='D', default=-1,
                            help="which encoder out layer for decoding")


    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        self.reencoder = self.build_reencoder(args)
 
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
        # missing_keys, unexpected_keys =  torch.nn.Module.load_state_dict(self, state_dict, strict=False)
        logger.info(f"pretrained model missing kyes | : {missing_keys}")
        logger.info(f"pretrained model unexpect kyes | : {unexpected_keys}")
        return missing_keys, unexpected_keys

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        # if args.pretrained_encoder_checkpoint is not None:
        #     logger.info(f"load pretrained model for encoder")
        #     model_state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_encoder_checkpoint, arg_overrides=None)["model"]
        #     keys=list(model_state.keys())
        #     for k in keys:
        #         if k.startswith('decoder.'):
        #             del model_state[k]
        #         if k.startswith('encoder.'):
        #             model_state[k[8:]] = model_state.pop(k)
        #     missing_keys, unexpect_keys = encoder.load_state_dict(model_state, strict=False)
        #     logger.info(f"pretrained encoder missing kyes | : {missing_keys}")
        #     logger.info(f"pretrained encoder unexpect kyes | : {unexpect_keys}")

        # if args.pretrained_decoder_checkpoint is not None:
        #     logger.info(f"load pretrained model for decoder")
        #     model_state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_decoder_checkpoint, arg_overrides=None)["model"]
        #     keys=list(model_state.keys())
        #     for k in keys:
        #         if k.startswith('encoder.'):
        #             del model_state[k]
        #         if k.startswith('decoder.'):
        #             model_state[k[8:]] = model_state.pop(k)
        #     missing_keys, unexpect_keys = decoder.load_state_dict(model_state, strict=False)
        #     logger.info(f"pretrained decoder missing kyes | : {missing_keys}")
        #     logger.info(f"pretrained decoder unexpect kyes | : {unexpect_keys}")
        
        
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if getattr(args, "mlm", False):
            encoder = TransformerEncoder(args, src_dict, embed_tokens)
        else:
            encoder = MLMTransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "load_pretrained_encoder_from", None):
            logger.info(f"load pretrained model for encoder")
            model_state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_encoder_from, arg_overrides=None)["model"]
            keys=list(model_state.keys())
            for k in keys:
                if k.startswith('decoder.'):
                    del model_state[k]
                if k.startswith('encoder.'):
                    model_state[k[8:]] = model_state.pop(k)
            missing_keys, unexpect_keys = encoder.load_state_dict(model_state, strict=False)
            logger.info(f"pretrained encoder missing kyes | : {missing_keys}")
            logger.info(f"pretrained encoder unexpect kyes | : {unexpect_keys}")

            # logger.info(f"load pretrained model for encoder")
            # encoder = checkpoint_utils.load_pretrained_component_from_model(
            #     component=encoder, checkpoint=args.load_pretrained_encoder_from
            # )
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder =  CLMTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

        if getattr(args, "load_pretrained_decoder_from", None):
            logger.info(f"load pretrained model for decoder")
            model_state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(args.load_pretrained_decoder_from, arg_overrides=None)["model"]
            keys=list(model_state.keys())
            for k in keys:
                if k.startswith('encoder.'):
                    del model_state[k]
                if k.startswith('decoder.'):
                    model_state[k[8:]] = model_state.pop(k)
            missing_keys, unexpect_keys = decoder.load_state_dict(model_state, strict=False)
            logger.info(f"pretrained decoder missing kyes | : {missing_keys}")
            logger.info(f"pretrained decoder unexpect kyes | : {unexpect_keys}")

            # logger.info(f"load pretrained model for decoder")
            # decoder = checkpoint_utils.load_pretrained_component_from_model(
            #     component=decoder, checkpoint=args.load_pretrained_decoder_from
            # )

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
            logger.info(f"not existing pretrained model for reencoder...")
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

        reencoder_out = self.reencoder(encoder_out["encoder_out"][0], encoder_out["encoder_padding_mask"][0])

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

    def forward_encoder(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = True,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
        )
        return encoder_out

    def forward_encoder_and_reencoder(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = True,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
        )
        reencoder_out = self.reencoder(encoder_out["encoder_out"][0], encoder_out["encoder_padding_mask"][0])
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
        reencoder_out = self.reencoder(encoder_out["encoder_out"][0], encoder_out["encoder_padding_mask"][0])
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


    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="subword_nmt",
        **kwargs,
    ):
        """
        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model
        file. Downloads and caches the pre-trained model file if needed.

        The base implementation returns a
        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to
        generate translations or sample from language models. The underlying
        :class:`~fairseq.models.FairseqModel` can be accessed via the
        *generator.models* attribute.

        Other models may override this to implement custom hub interfaces.

        Args:
            model_name_or_path (str): either the name of a pre-trained model to
                load or a path/URL to a pre-trained model state dict
            checkpoint_file (str, optional): colon-separated list of checkpoint
                files in the model archive to ensemble (default: 'model.pt')
            data_name_or_path (str, optional): point args.data to the archive
                at the given path/URL. Can start with '.' or './' to reuse the
                model archive path.
        """
        from fairseq import hub_utils
        from .monmt_interface import MoNMTHubInterface

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            **kwargs,
        )
        logger.info(x["args"])
        return MoNMTHubInterface(x["args"], x["task"], x["models"][0])

    @classmethod
    def hub_models(cls):
        return {}

@register_model_architecture("reencoder_transformer", "reencoder_transformer_base")
def modular_base_architecture(args):

    args.reencoder_layers = getattr(args, "reencoder_layers", 6)
    args.reencoder_embed_dim = getattr(args, "reencoder_embed_dim", 512)
    args.reencoder_ffn_embed_dim = getattr(args, "reencoder_ffn_embed_dim", 4 * 512)
    args.reencoder_attention_heads = getattr(args, "reencoder_attention_heads", 8)
    args.reencoder_normalize_before = getattr(args, "reencoder_normalize_before", False)
    args.reencoder_layerdrop = getattr(args, "reencoder_layerdrop", 0)
    args.reencoder_layers_to_keep = getattr(args, "reencoder_layers_to_keep", None)
    base_architecture(args)


@register_model_architecture("reencoder_transformer", "reencoder_transformer_big")
def modular_big_architecture(args):

    args.reencoder_layers = getattr(args, "reencoder_layers", 6)
    args.reencoder_embed_dim = getattr(args, "reencoder_embed_dim", 1024)
    args.reencoder_ffn_embed_dim = getattr(args, "reencoder_ffn_embed_dim", 4 * 1024)
    args.reencoder_attention_heads = getattr(args, "reencoder_attention_heads", 16)
    args.reencoder_normalize_before = getattr(args, "reencoder_normalize_before", False)
    args.reencoder_layerdrop = getattr(args, "reencoder_layerdrop", 0)
    args.reencoder_layers_to_keep = getattr(args, "reencoder_layers_to_keep", None)
    transformer_vaswani_wmt_en_fr_big(args)