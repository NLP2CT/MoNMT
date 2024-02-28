generate_f(){

s=$1
t=$2
mlm=$3
clm=$4
srcdae=$5
tgtdae=$6
encls=$7
decls=$8
recls=$9
recdim=${10}
recffndim=${11}
rechead=${12}
monopath=${13}


rootdir=$userpjh
langs=$s,$t
ptlangdirection=${s}2${t}
if [[ $t = 'en' && $mlm = '0' && $clm = '0' && $srcdae = '1' && $tgtdae = '1' ]]; then
    ptlangdirection=${t}2${s}
    langs=$t,$s
fi
echo $langs
echo $ptlangdirection

ptmodel=$rootdir/modular-translation/wmt14ende/daept/wmt14deenbpe/big/en2de/transformerlgid_enc_${encls}_dec_${decls}_catt_0_${mlm}_${clm}_${srcdae}_${tgtdae}_w4kmt8192dp01_${monopath}_mt__mlm_lm_srcdae_tgtdae_ft_bt/checkpoints/checkpoint_last.pt


langdirection=${s}2${t}
datadir=$rootdir/data/mt-parallel-corpus/wmt14deen/databin


subtask=translation_300ksteps_enc_${encls}_dec_${decls}_recls_${recls}_recdim_${recdim}_ptmono_${monopath}
logdir=./daeptMoNMT/big/$langdirection/$subtask/tensorboardlogs
savedir=./daeptMoNMT/big/$langdirection/$subtask/checkpoints

mkdir -p $logdir
mkdir -p $savedir

python -m torch.distributed.launch --nproc_per_node=4 ./train.py \
    $datadir \
    --fp16 \
    --user-dir ../fairsequserdir \
    --upsample-primary 1 \
    --update-freq 1 \
    --patience 20 \
    --task translation_with_reencoder_task \
    --langs $langs \
    --finetune-from-model $ptmodel \
    --freeze-params "(.embed.)|(\bencoder\b.)|(decoder.)" \
    --add-lang-symbol --add-mask-symbol \
    --append-source-id  --share-dictionary \
    --arch reencoder_transformer_big --share-all-embeddings \
    --encoder-layers $encls --decoder-layers $decls \
    --reencoder-layers $recls \
    --reencoder-embed-dim $recdim \
    --reencoder-ffn-embed-dim $recffndim \
    --reencoder-attention-heads $rechead \
    -s ${s} -t ${t} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.1 --weight-decay 0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --max-update 300000 \
    --save-dir $savedir \
    --save-interval-updates 1000 --keep-interval-updates 1 \
    --no-epoch-checkpoints \
    --tensorboard-logdir $logdir \
    --no-progress-bar --log-format simple --log-interval 50 \
    --ddp-backend no_c10d \
    | tee -a $logdir/train.log
    # --eval-bleu \
    # --eval-bleu-args '{"beam": 4, "lenpen": 0.6}' \
    # --eval-bleu-detok moses \
    # --eval-bleu-remove-bpe \
    # --eval-bleu-print-samples \

ckpt=checkpoint_best.pt
echo "run generating...."
rm $logdir/${s}_${t}.$ckpt.gen
if [ -f $logdir/${s}_${t}.$ckpt.gen ]; then
    echo "found results.......!!"
else

    fairseq-generate \
        $datadir \
        -s ${s} -t ${t} \
        --user-dir ./fairsequserdir \
        --task translation_with_reencoder_task \
        --langs $langs \
        --add-lang-symbol --add-mask-symbol \
        --append-source-id  --share-dictionary \
        --path $savedir/$ckpt \
        --beam 5 --lenpen 1 --remove-bpe > $logdir/${s}_${t}.$ckpt.gen \

fi

bleuscript=./scripts
bash $bleuscript/sacrebleu.sh wmt14/full ${s} ${t} $logdir/${s}_${t}.$ckpt.gen | tee -a $logdir/test.log

bash $bleuscript/compound_split_bleu.sh $logdir/${s}_${t}.$ckpt.gen | tee -a $logdir/test.log

}

s=$1
t=$2
encls=$3
decls=$4
recls=$5
recdim=$6
recffndim=${7}
rechead=${8}
monopath=${9}



generate_f $s $t 0 0 1 1 $encls $decls $recls $recdim $recffndim $rechead $monopath

