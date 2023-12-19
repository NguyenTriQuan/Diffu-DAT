data_name=iwslt14
data_dir=data-bin/iwslt14.tokenized.de-en
arch=glat_decomposed_link_base
criterion=ard_dag_loss
checkpoint_dir=/cm/archive/quannt40/Diffu-DAT/checkpoints_1/$data_name"_"$arch"_"$criterion
# checkpoint_dir=/cm/archive/quannt40/Diffu-DAT/checkpoints/$data_name"_"$arch

average_checkpoint_path=$checkpoint_dir"/average.pt"

fairseq-generate $data_dir  \
    --path $average_checkpoint_path  --beam 5 --lenpen 0.6 \
    --user-dir fs_plugins --task translation_lev_modified \
    --remove-bpe --max-tokens 4096 --seed 0 \
    --gen-subset train  > data-bin/iwslt14/distill_full_0.txt


python examples/backtranslation/extract_bt_data.py --minlen 1 --maxlen 4096 --ratio 3 \
    --output data-bin/iwslt14 --srclang de --tgtlang en data-bin/iwslt14/distill_full_0.txt
