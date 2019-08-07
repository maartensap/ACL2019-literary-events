if [ ! -d ../data/bert/ ]
then
    mkdir ../data/bert/
fi

for split in train dev test
do
    for i in `ls ../data/tsv/${split}/`
    do
        if [ ! -d ../data/bert/${split}/ ]
        then
            mkdir ../data/bert/${split}/
        fi
        echo ${split}/$i
        python return_bert_features.py --file ../data/tsv/${split}/$i --model_path=bert-base-cased --output=../data/bert/${split}/$i
    done
done

exit

for i in `ls ../data/tsv/dev/`
do echo $i
python return_bert_features.py --file ../data/tsv/dev/$i --model_path=bert-base-cased --output=../data/bert/dev/$i
done

for i in `ls ../data/tsv/test/`
do echo $i
python return_bert_features.py --file ../data/tsv/test/$i --model_path=bert-base-cased --output=../data/bert/test/$i
done
