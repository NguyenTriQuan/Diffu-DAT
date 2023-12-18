input_dir=Diffu-DAT/data-bin/wmt14_ende        # directory of raw text data
data_dir=Diffu-DAT/data-bin/wmt14_ende   # directory of the generated binarized data
src=en                            # source language id
tgt=de                            # target language id
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${input_dir}/train.${src}-${tgt} --validpref ${input_dir}/valid.${src}-${tgt} --testpref ${input_dir}/test.${src}-${tgt} \
    --src-dict ${input_dir}/dict.${src}.txt --tgt-dict ${input_dir}/dict.${tgt}.txt \
    --destdir ${data_dir} --workers 32