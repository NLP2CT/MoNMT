#!/bin/bash

set -eu

rootdir=$userpjh
MOSES=$rootdir/data/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
REMOVE_DIACRITICS=$rootdir/data/wmt16-scripts/preprocess/remove-diacritics.py
NORMALIZE_ROMANIAN=$rootdir/data/wmt16-scripts/preprocess/normalise-romanian.py
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl

sys=$1
ref=$2

lang=ro
for file in $sys $ref; do
  cat $file \
  | $REPLACE_UNICODE_PUNCT \
  | $NORM_PUNC -l $lang \
  | $REM_NON_PRINT_CHAR \
  | $NORMALIZE_ROMANIAN \
  | $REMOVE_DIACRITICS \
  | $TOKENIZER -no-escape -l $lang \
  > $(basename $file).tok
done

cat $(basename $sys).tok | sacrebleu -tok none -s none -b $(basename $ref).tok