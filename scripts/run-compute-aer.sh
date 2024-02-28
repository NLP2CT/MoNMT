calculate_shift_att_AER(){   
    src=$1
    tgt=$2
    modeldir=$3
    rootdir=/mnt/nas/users/pangjianhui.pjh
    bpe=$rootdir/data/wordalignments/DeEn/bpe  ## put the BPE files here
    databin=$rootdir/data/wordalignments/DeEn/bpe/databin
    resdir=$databin/result/${src}2${tgt}_vanilla  && mkdir -p $resdir 
    ref_align=$rootdir/data/wordalignments/DeEn/alignmentDeEn.talp  # define your reference alignment file here

    split='test'
    echo "start to extract ${src}2${tgt} alignment."
    python generate_align.py $databin -s $src -t $tgt \
        --path $modeldir/checkpoint_best.pt --user-dir  $rootdir/fairsequserdir \
        --max-tokens 4000 --beam 1 --remove-bpe --quiet --print-vanilla-alignment --decoding-path $resdir  \
        --alignment-task vanilla --gen-subset $split --alignment-layer 2 --set-shift 
    
    python scripts/aer/sentencepiece_to_word_alignments.py --src $bpe/test.$src \
       --tgt $bpe/test.$tgt --input $resdir/$split.${src}2${tgt}.align --output $resdir/$split.${src}2${tgt}.raw.align


    echo "=====calculate AER for shift-att method=========="   
    python scripts/aer/aer.py ${ref_align} $resdir/$split.${src}2${tgt}.raw.align --fAlpha 0.5 --oneRef 
    # python scripts/aer/aer.py ${ref_align} $resdir/$split.${tgt}2${src}.raw.align --fAlpha 0.5 --oneRef --reverseHyp
    # python scripts/aer/aer.py ${ref_align} $resdir/$split.${src}2${tgt}.bidir.align --fAlpha 0.5 --oneRef 
}

fumodeldir=runstation/wmt14deen100k/checkpoints/en2de/transformerbaselgid_1_0_0_0_0_0_1_w4kmt8192dp01_5m_256_mono_mt__mlm_lm_srcdae_tgtdae_ft_bt
rootdir=/mnt/nas/users/pangjianhui.pjh
modeldir=$rootdir/Graformer-local/$fumodeldir
calculate_shift_att_AER en de $modeldir