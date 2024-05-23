dataset=./amr_data/amr_2.0_politifact/csqa
python3 ./prepare/extract_vocab.py --train_data  ${dataset}/train.pred.txt 
                   --amr_files ${dataset}/train.pred.txt ${dataset}/dev.pred.txt ${dataset}/test.pred.txt \
                   --nprocessors 8
mv *_vocab ${dataset}/.

#python3 extract_property.py --train_data ${dataset}/train.pred.txt \
#                   --amr_files ${dataset}/train.pred.txt ${dataset}/dev.pred.txt ${dataset}/test.pred.txt \
#                   --nprocessors 8 --extend True --concept_seed question_amr
#${dataset}/train.pred.txt --dev_data --test_data ${dataset}/test.pred.txt\
