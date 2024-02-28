# MoNMT
MoNMT: Modularly Leveraging Monolingual and Bilingual Knowledge for Neural Machine Translation

# Environment
- The code is based on the [fairseq](https://github.com/facebookresearch/fairseq) toolkit, version 1.0.0a0, forked from [Graformer](https://github.com/sunzewei2715/Graformer) codebase.
- python 3.8

# Training
We show one example for training a MoNMT model.

## Pretrain
See run-pretrian.sh
- After training, we can get a Encoder-to-Decoder Denoising model for the source and the target languages.
- In this example, the source and target languages share the same encoding and decoding modules.

## train MoNMT
See run-train-MoNMT.sh
- After training, we can get a source-to-target MoNMT translation model.

# Other Information
- The paper has been accepted by LREC-COLING 2024.