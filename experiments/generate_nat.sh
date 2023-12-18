data_name=iwslt14
data_dir=data-bin/iwslt14.tokenized.de-en
arch=glat_decomposed_link_base
criterion=ard_dag_loss
checkpoint_dir=/cm/archive/quannt40/Diffu-DAT/checkpoints_1/$data_name"_"$arch"_"$criterion
average_checkpoint_path=$checkpoint_dir"/average.pt"

# python3 ./fs_plugins/scripts/average_checkpoints.py --inputs ${checkpoint_dir} \
#                 --max-metric --best-checkpoints-metric bleu --num-best-checkpoints-metric 5 --output ${average_checkpoint_path}


fairseq-generate ${data_dir} \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 1 \
    --remove-bpe --max-tokens 4096 --seed 0 \
    --decode-strategy lookahead \
    --model-overrides "{\"decode_strategy\":\"lookahead\",\"decode_upsample_scale\":0.1,\"decode_beta\":1}" \
    --skip-invalid-size-inputs-valid-test
