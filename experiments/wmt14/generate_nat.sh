data_name=wmt14
data_dir=data-bin/wmt14_ende
arch=glat_decomposed_link_base
criterion=ard_dag_loss
checkpoint_dir=/cm/archive/quannt40/Diffu-DAT/wmt14/$data_name"_"$arch"_"$criterion

average_checkpoint_path=$checkpoint_dir"/average.pt"

python3 ./fs_plugins/scripts/average_checkpoints.py --inputs ${checkpoint_dir} \
                --max-metric --best-checkpoints-metric bleu --num-best-checkpoints-metric 5 --output ${average_checkpoint_path}


CUDA_VISIBLE_DEVICES=7 fairseq-generate ${data_dir} \
    --gen-subset test --user-dir fs_plugins --task translation_lev_modified \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --beam 5 \
    --remove-bpe --max-tokens 4096 --seed 0 \
    --path ${average_checkpoint_path} \
    --skip-invalid-size-inputs-valid-test \
    --decode-upsample-scale 8.0 \
    --model-overrides "{\"decode_strategy\":\"jointviterbi\",\"decode_viterbibeta\":1}" \

