#!/bin/bash

rootdir=$userpjh
MOSES=$rootdir/data/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
REMOVE_DIACRITICS=$rootdir/data/wmt16-scripts/preprocess/remove-diacritics.py
NORMALIZE_ROMANIAN=$rootdir/data/wmt16-scripts/preprocess/normalise-romanian.py
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl

if [ $# -ne 1 ]; then
    echo "usage: $0 GENERATE_PY_OUTPUT"
    exit 1
fi

GEN=$1

SYS=$GEN.spm.sys
REF=$GEN.spm.ref

if [ $(tail -n 1 $GEN | grep BLEU | wc -l) -ne 1 ]; then
    echo "not done generating"
    exit
fi

# | perl $REPLACE_UNICODE_PUNCT -l de \

grep ^D $GEN | cut -f3- \
    | perl $TOKENIZER -no-escape -l de \
    | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $SYS

grep ^T $GEN | cut -f2- \
    | perl $TOKENIZER -no-escape -l de \
    | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $REF

fairseq-score --sys $SYS --ref $REF
# perl $MOSES/scripts/generic/multi-bleu.perl $REF < $SYS
