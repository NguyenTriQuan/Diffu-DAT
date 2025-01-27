data_name=iwslt14
data_dir=data-bin/iwslt14.tokenized.de-en
arch=transformer_iwslt_de_en_base
criterion=label_smoothed_cross_entropy
checkpoint_dir=/cm/archive/quannt40/Diffu-DAT/checkpoints_1/$data_name"_"$arch"_"$criterion
# checkpoint_dir=/cm/archive/quannt40/Diffu-DAT/checkpoints/$data_name"_"$arch

average_checkpoint_path=$checkpoint_dir"/average.pt"

python3 ./fs_plugins/scripts/average_checkpoints.py --inputs ${checkpoint_dir} \
                --max-metric --best-checkpoints-metric bleu --num-best-checkpoints-metric 5 --output ${average_checkpoint_path}


fairseq-generate ${data_dir} \
    --gen-subset test --user-dir fs_plugins \
    --beam 1 \
    --remove-bpe --max-tokens 4096 --seed 0 \
    --path ${average_checkpoint_path} \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu-print-samples \ 
    # --task translation_lev_modified
