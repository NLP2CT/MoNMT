# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
# ================================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from typing import List
from silkflow_framework.sdk.base_pipeline import BasePipeline
from silkflow_framework.sdk.expt_manager import ExptManager


class PipelineTemplate(BasePipeline):
    """A pipeline template to run train"""

    def __init__(self, expt_manager: ExptManager, prefix: str, dep_ops: List = [], configs: dict = {},
                 params: dict = {}):
        if not prefix:
            prefix = self.__class__.__name__
        default_params = {}
        # update experiment directory.
        silkflow_detail_dir = '%s/silkflow_detail' % (os.getcwd() if expt_manager is None else expt_manager.expt_dir)
        if not expt_manager or not hasattr(expt_manager, '_update_runtime_status'):
            expt_manager = ExptManager(expt_dir=silkflow_detail_dir)
        else:
            expt_manager._update_runtime_status(expt_dir=silkflow_detail_dir)
        # init base pipeline.
        super()._init(expt_manager, prefix, dep_ops=dep_ops, configs=configs, params=params,
                      default_params=default_params)

    def _define(self):
        action_op = super()._add_op(name="Graformer-local-20220713-11_03_50",
                                    image="reg.docker.alibaba-inc.com/silkflow/pytorch:1.7-cuda10.1-cudnn7-fairseq-apex-fix",
                                    command="cd /mnt/nas/users/pangjianhui.pjh/Graformer-local;bash run-compute-aer-mlmdae.sh",
                                    gpus=1,
                                    cpu=12,
                                    memory=64,
                                    requirements="",
                                    node_selector={'sigma.ali/node-sn':'qtfcu19490004'})
        self.last_ops += action_op