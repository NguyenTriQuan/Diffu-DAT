# input_dir=data-bin/wmt14_ende        # directory of raw text data
# data_dir=data-bin/wmt14_ende   # directory of the generated binarized data
# src=en                            # source language id
# tgt=de                            # target language id
# fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
#     --trainpref ${input_dir}/train.${src}-${tgt} --validpref ${input_dir}/valid.${src}-${tgt} --testpref ${input_dir}/test.${src}-${tgt} \
#     --joined-dictionary \
#     --destdir ${data_dir} --workers 32


input_dir=data-bin/wmt14_ende        # directory of pre-processed text data
data_dir=data-bin/wmt14_ende   # directory of the generated binarized data
src=src                           # source suffix
tgt=tgt                           # target suffix

# The following command require files:
#     train.${src} train.${tgt} valid.${src} valid.${tgt} test.${src} test.${tgt}
#     dict.${src}.txt  dict.${tgt}.txt

fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.${src}-${tgt} --validpref ${input_dir}/valid.${src}-${tgt} --testpref ${input_dir}/test.${src}-${tgt} \
    --src-dict ${input_dir}/dict.${src}.txt --tgt-dict ${input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 32 --user-dir fs_plugins

# [--seg-tokens 32] is optional, it should be set when you use pre-trained models; otherwise, just remove it.