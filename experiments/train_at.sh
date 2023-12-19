data_name=iwslt14
data_dir=data-bin/iwslt14.tokenized.de-en
arch=transformer_iwslt_de_en_base
criterion=label_smoothed_cross_entropy
checkpoint_dir=/cm/archive/quannt40/Diffu-DAT/checkpoints_1/$data_name"_"$arch"_"$criterion


CUDA_VISIBLE_DEVICES=1 fairseq-train ${data_dir}  \
    --user-dir fs_plugins \
    --arch ${arch} --share-decoder-input-output-embed \
    --activation-fn gelu \
    --criterion ${criterion} \
    --optimizer adam --adam-betas '(0.9,0.999)' --fp16 \
    --label-smoothing 0.1 --weight-decay 0.01 --dropout 0.3 \
    --lr-scheduler inverse_sqrt  --warmup-updates 30000   \
    --clip-norm 0.1 --lr 0.0005 --warmup-init-lr '1e-07' --stop-min-lr '1e-09' \
    --max-tokens 4096  --update-freq 1 --grouped-shuffling \
    --max-update 200000 --max-tokens-valid 4096 \
    --save-interval 1  --save-interval-updates 10000  \
    --seed 0 \
    --valid-subset test \
    --validate-interval 1 --validate-interval-updates 10000 \
    --eval-bleu --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --skip-invalid-size-inputs-valid-test \
    --fixed-validation-seed 7 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric   \
    --keep-best-checkpoints 5 --save-dir ${checkpoint_dir} \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --log-format 'simple' --log-interval 100 \
    --wandb-project Diffu-DAT