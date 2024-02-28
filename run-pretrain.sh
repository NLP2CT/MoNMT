

generate_f(){

s=$1
t=$2
lambda0=$3
lambda1=$4
lambda2=$5
lambda3=$6
lambda4=$7
lambda5=$8
lambda6=$9
monopath=${10}

paratask=(
    # "wmt14deen10k"
    # "wmt14deen50k"
    # "wmt14deen100k"
    # "wmt14deenv9news200k"
    # "wmt14deen500k"
    "wmt14deen"
    # "wmt14deen1m"
    # "wmt14deen2m"
)

mono=5m_256_mono

rootdir=$userpjh

for para in "${paratask[@]}"; do


btrootpath=/mnt/nas/users/pangjianhui.pjh/data/mt-parallel-corpus/wmt14deen/bpe/wmt14_en_de_scaling18/20m-256-mono/20mpiece/de/backtranslation
btpath=$btrootpath/run${para}/$monopath

datadir=$rootdir/data/mt-parallel-corpus/wmt14deen/databin

langdirection=${s}2${t}
subtask=transformerlgid_enc_${lambda5}_dec_${lambda6}_catt_${lambda0}_${lambda1}_${lambda2}_${lambda3}_${lambda4}_w4kmt8192dp01_${monopath}_mt__mlm_lm_srcdae_tgtdae_ft_bt
logdir=./newnas/daept/wmt14deenbpe/big/$langdirection/$subtask/tensorboardlogs
savedir=./newnas/daept/wmt14deenbpe/big/$langdirection/$subtask/checkpoints
mkdir -p $logdir
mkdir -p $savedir


python -m torch.distributed.launch --nproc_per_node=4 ./train.py \
    $datadir \
    --fp16 \
    --monopath $monopath \
    --upsample-primary 1 \
    --update-freq 2 \
    --task multitasktranslationalllgid \
    --mlm \
    --add-lang-symbol --append-source-id --add-mask-symbol --lang-tok-first-add \
    --share-dictionary \
    --arch mlm_lm_transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --encoder-layers ${lambda5} --decoder-layers ${lambda6} \
    -s ${s} -t ${t} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.1 --weight-decay 0 \
    --criterion label_smoothed_cross_entropy_with_all --label-smoothing 0.1 \
    --max-tokens 8192 \
    --lm-tokens-per-sample 512 \
    --lm-sample-break-mode "eos" \
    --mlm-tokens-per-sample 512 \
    --mlm-sample-break-mode "eos" \
    --lambda-mt-config "${lambda0}" \
    --lambda-mlm-config ${lambda1} \
    --lambda-lm-config ${lambda2} \
    --lambda-srcdae-config ${lambda3} \
    --lambda-tgtdae-config ${lambda4} \
    --max-update 200000 \
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

# ckpt=checkpoint_best.pt
# echo "run generating...."
# rm $logdir/${s}_${t}.$ckpt.gen
# if [ -f $logdir/${s}_${t}.$ckpt.gen ]; then
#     echo "found results.......!!"
# else

#     fairseq-generate \
#         $datadir \
#         -s ${s} -t ${t} \
#         --task multitasktranslationalllgid \
#         --share-dictionary --append-source-id \
#         --testing \
#         --path $savedir/$ckpt \
#         --beam 4 --lenpen 0.6 --remove-bpe > $logdir/${s}_${t}.$ckpt.gen \

# fi

# bleuscript=./Graformer-local/scripts
# #bash $bleuscript/sacrebleu.sh wmt14/full ${s} ${t} $logdir/${s}_${t}.$ckpt.gen | tee -a $logdir/test.log

# bash $bleuscript/compound_split_bleu.sh $logdir/${s}_${t}.$ckpt.gen | tee -a $logdir/test.log


done
}

src=$1
tgt=$2
encoderlayers=$3
decoderlayers=$4
mono=$5
generate_f $src $tgt 0 0 0 1 1 $encoderlayers $decoderlayers $mono 