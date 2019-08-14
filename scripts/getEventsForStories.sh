BEST_MODEL=../results/bert/bert.3.hdf5
for d in imagined
do
    mkdir ../results/bert/$d/
    for i in `ls ../data/bert/${d}`
    do
	# python neural.py --existingModel $BEST_MODEL --predictionsFile ../results/bert/$d/$i --fileToPredict ../data/bert/$d/$i --mode predict -e ../data/guten.vectors.txt --bert
        echo hi
    done
done
# exit
# Exporting stories
python exportStories2Tsv.py --input_file ~/research/memoryLang/processedFiles/pilotV3-11_imagined.csv --output_dir ../data/tsv/imagined/
python exportStories2Tsv.py --input_file ~/research/memoryLang/processedFiles/pilotV3-11_recalled.csv --output_dir ../data/tsv/recalled/

# Getting BERT features
for d in imagined recalled
do
    for i in `ls ../data/tsv/${d}`
    do
        if [ ! -d ../data/bert/${d}/ ]
        then
            mkdir ../data/bert/${d}/
        fi
        python return_bert_features.py --file ../data/tsv/${d}/$i --model_path=bert-base-cased --output=../data/bert/${d}/$i
    done
done

# Run prediction (NOTE: need to have a model before you can run this)

# Tag events with best model
BEST_MODEL=../results/bert/bert.3.hdf5

for d in imagined recalled
do
    mkdir ../results/bert/$d/
    for i in `ls ../data/bert/${d}`
    do
	python neural.py --existingModel $BEST_MODEL --predictionsFile ../results/bert/$d/$i --fileToPredict ../data/bert/$d/$i --mode predict -e ../data/guten.vectors.txt --bert
    done
done
