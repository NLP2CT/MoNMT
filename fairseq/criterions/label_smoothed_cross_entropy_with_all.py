# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from fairseq import metrics, utils, modules
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class LabelSmoothedCrossEntropyWithALLCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy_with_all", dataclass=LabelSmoothedCrossEntropyWithALLCriterionConfig
)
class LabelSmoothedCrossEntropyWithALLCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        tpu=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.tpu = tpu

    def forward(self, model, sample, reduce=True, mode='mt'):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert mode in ['mt', 'mlm', 'lm', 'dae', 'srcdae', 'tgtdae', 'ft', 'bt']
        if mode in ['mt', 'srcdae', 'dae', 'tgtdae', 'ft', 'bt']:
            loss, sample_size, logging_output = self.forward_compute_mt_loss(model, sample, reduce=reduce)
        if mode == "mlm":
            loss, sample_size, logging_output = self.forward_compute_mlm_loss(model, sample, reduce=reduce)
        if mode == "lm":
            loss, sample_size, logging_output = self.forward_compute_lm_loss(model, sample, reduce=reduce)

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def forward_compute_mt_loss(self, model, sample, reduce=True):
        # print('mt: ', sample["net_input"])
        # print('mt: ', sample['target'])
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
    
    def forward_compute_mlm_loss(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample["target"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )
        # print('mlm: ', sample["net_input"])
        # print('mlm: ', sample['target'])
        logits = model(**sample["net_input"], masked_tokens=masked_tokens)['mlm_out']
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output
    
    def forward_compute_lm_loss(self, model, sample, reduce=True):
        # print('lm: ', sample["net_input"])
        # print('lm: ', sample['target'])
        net_output = model.forward_lm_layers(**sample["net_input"], only_return_lm_output=True)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, net_output).view(-1)
        # loss = F.nll_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduction="sum" if reduce else "none"
        # )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["lm_n_correct"] = utils.item(n_correct.data)
            logging_output["lm_total"] = utils.item(total.data)
        return loss, sample_size, logging_output


    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training with mlm with lm."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt:loss", 0) for log in logging_outputs)
        mlm_loss_sum = sum(log.get("mlm:loss", 0) for log in logging_outputs)
        lm_loss_sum = sum(log.get("lm:loss", 0) for log in logging_outputs)
        mt_nll_loss_sum = sum(log.get("mt:nll_loss", 0) for log in logging_outputs)
        lm_nll_loss_sum = sum(log.get("lm:nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        mt_ntokens = sum(log.get("mt:ntokens", 0) for log in logging_outputs)
        lm_ntokens = sum(log.get("lm:ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        mt_sample_size = sum(log.get("mt:sample_size", 0) for log in logging_outputs)
        mlm_sample_size = sum(log.get("mlm:sample_size", 0) for log in logging_outputs)
        lm_sample_size = sum(log.get("lm:sample_size", 0) for log in logging_outputs)     
    

        mt_loss = mt_loss_sum / mt_sample_size / math.log(2) if mt_sample_size > 0.0 else 0.0
        mlm_loss = mlm_loss_sum / mlm_sample_size / math.log(2) if mlm_sample_size > 0.0 else 0.0
        lm_loss = lm_loss_sum / lm_sample_size / math.log(2)if lm_sample_size > 0.0 else 0.0

        dae_loss_sum = sum(log.get("dae:loss", 0) for log in logging_outputs)
        dae_nll_loss_sum = sum(log.get("dae:nll_loss", 0) for log in logging_outputs)
        dae_ntokens = sum(log.get("dae:ntokens", 0) for log in logging_outputs)
        dae_sample_size = sum(log.get("dae:sample_size", 0) for log in logging_outputs)
        dae_loss = dae_loss_sum / dae_sample_size / math.log(2) if dae_sample_size > 0.0 else 0.0

        srcdae_loss_sum = sum(log.get("srcdae:loss", 0) for log in logging_outputs)
        srcdae_nll_loss_sum = sum(log.get("srcdae:nll_loss", 0) for log in logging_outputs)
        srcdae_ntokens = sum(log.get("srcdae:ntokens", 0) for log in logging_outputs)
        srcdae_sample_size = sum(log.get("srcdae:sample_size", 0) for log in logging_outputs)
        srcdae_loss = srcdae_loss_sum / srcdae_sample_size / math.log(2) if srcdae_sample_size > 0.0 else 0.0


        tgtdae_loss_sum = sum(log.get("tgtdae:loss", 0) for log in logging_outputs)
        tgtdae_nll_loss_sum = sum(log.get("tgtdae:nll_loss", 0) for log in logging_outputs)
        tgtdae_ntokens = sum(log.get("tgtdae:ntokens", 0) for log in logging_outputs)
        tgtdae_sample_size = sum(log.get("tgtdae:sample_size", 0) for log in logging_outputs)
        tgtdae_loss = tgtdae_loss_sum / tgtdae_sample_size / math.log(2) if tgtdae_sample_size > 0.0 else 0.0

        ft_loss_sum = sum(log.get("ft:loss", 0) for log in logging_outputs)
        ft_nll_loss_sum = sum(log.get("ft:nll_loss", 0) for log in logging_outputs)
        ft_ntokens = sum(log.get("ft:ntokens", 0) for log in logging_outputs)
        ft_sample_size = sum(log.get("ft:sample_size", 0) for log in logging_outputs)
        ft_loss = ft_loss_sum / ft_sample_size / math.log(2) if ft_sample_size > 0.0 else 0.0

        bt_loss_sum = sum(log.get("bt:loss", 0) for log in logging_outputs)
        bt_nll_loss_sum = sum(log.get("bt:nll_loss", 0) for log in logging_outputs)
        bt_ntokens = sum(log.get("bt:ntokens", 0) for log in logging_outputs)
        bt_sample_size = sum(log.get("bt:sample_size", 0) for log in logging_outputs)
        bt_loss = bt_loss_sum / bt_sample_size / math.log(2) if bt_sample_size > 0.0 else 0.0

        metrics.log_scalar(
            "loss", mt_loss+mlm_loss+lm_loss+srcdae_loss+dae_loss+tgtdae_loss+ft_loss+bt_loss, sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss, mt_sample_size, round=3
        )
        metrics.log_scalar(
            "mt_nll_loss", mt_nll_loss_sum / mt_ntokens / math.log(2) if mt_ntokens > 0.0 else 0.0, mt_ntokens, round=3
        )
        metrics.log_derived(
            "mt_ppl", lambda meters: utils.get_perplexity(meters["mt_nll_loss"].avg)
        )
        metrics.log_scalar(
            "ft_loss", ft_loss, tgtdae_sample_size, round=3
        )
        metrics.log_scalar(
            "ft_nll_loss", ft_nll_loss_sum / ft_ntokens / math.log(2) if ft_ntokens > 0.0 else 0.0, ft_ntokens, round=3
        )
        metrics.log_derived(
            "ft_ppl", lambda meters: utils.get_perplexity(meters["ft_nll_loss"].avg)
        )
        metrics.log_scalar(
            "bt_loss", bt_loss, tgtdae_sample_size, round=3
        )
        metrics.log_scalar(
            "bt_nll_loss", bt_nll_loss_sum / bt_ntokens / math.log(2) if bt_ntokens > 0.0 else 0.0, bt_ntokens, round=3
        )
        metrics.log_derived(
            "bt_ppl", lambda meters: utils.get_perplexity(meters["bt_nll_loss"].avg)
        )

        metrics.log_scalar(
            "dae_loss", dae_loss, dae_sample_size, round=3
        )
        metrics.log_scalar(
            "dae_nll_loss", dae_nll_loss_sum / dae_ntokens / math.log(2) if dae_ntokens > 0.0 else 0.0, dae_ntokens, round=3
        )
        metrics.log_derived(
            "dae_ppl", lambda meters: utils.get_perplexity(meters["dae_nll_loss"].avg)
        )

        metrics.log_scalar(
            "srcdae_loss", srcdae_loss, srcdae_sample_size, round=3
        )
        metrics.log_scalar(
            "srcdae_nll_loss", srcdae_nll_loss_sum / srcdae_ntokens / math.log(2) if srcdae_ntokens > 0.0 else 0.0, srcdae_ntokens, round=3
        )
        metrics.log_derived(
            "srcdae_ppl", lambda meters: utils.get_perplexity(meters["srcdae_nll_loss"].avg)
        )

        metrics.log_scalar(
            "tgtdae_loss", tgtdae_loss, tgtdae_sample_size, round=3
        )
        metrics.log_scalar(
            "tgtdae_nll_loss", tgtdae_nll_loss_sum / tgtdae_ntokens / math.log(2) if tgtdae_ntokens > 0.0 else 0.0, tgtdae_ntokens, round=3
        )
        metrics.log_derived(
            "tgtdae_ppl", lambda meters: utils.get_perplexity(meters["tgtdae_nll_loss"].avg)
        )

        metrics.log_scalar(
            "mlm_loss", mlm_loss, mlm_sample_size, round=3
        )
        
        metrics.log_scalar(
            "lm_loss", lm_loss, lm_sample_size, round=3
        )
        metrics.log_scalar(
            "lm_nll_loss", lm_nll_loss_sum / lm_ntokens / math.log(2) if lm_ntokens > 0.0 else 0.0, lm_ntokens, round=3
        )
        metrics.log_derived(
            "lm_ppl", lambda meters: utils.get_perplexity(meters["lm_nll_loss"].avg)
        )

        

   
      
   
        
        # metrics.log_derived(
        #     "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        # )
        
        
        
        

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        lm_total = utils.item(sum(log.get("lm_total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("lm_total", total)
            n_correct = utils.item(
                sum(log.get("lm_n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("lm_n_correct", n_correct)
            metrics.log_derived(
                "lm_accuracy",
                lambda meters: round(
                    meters["lm_n_correct"].sum * 100.0 / meters["lm_total"].sum, 3
                )
                if meters["lm_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
