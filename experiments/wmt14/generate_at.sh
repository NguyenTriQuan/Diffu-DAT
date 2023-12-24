data_name=wmt14
data_dir=data-bin/wmt14_ende
arch=glat_decomposed_link_base
criterion=ard_dag_loss
checkpoint_dir=/cm/archive/quannt40/Diffu-DAT/wmt14/$data_name"_"$arch"_"$criterion

average_checkpoint_path=$checkpoint_dir"/average.pt"

python3 ./fs_plugins/scripts/average_checkpoints.py --inputs ${checkpoint_dir} \
                --max-metric --best-checkpoints-metric bleu --num-best-checkpoints-metric 5 --output ${average_checkpoint_path}


fairseq-generate ${data_dir} \
    --gen-subset test --user-dir fs_plugins \
    --beam 1 \
    --remove-bpe --max-tokens 4096 --seed 0 \
    --path ${average_checkpoint_path} \
    --skip-invalid-size-inputs-valid-test \
    # --task translation_lev_modified
