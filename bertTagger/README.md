# BERT realis tagger

To train a model based on the data using *hyperparameter* search over number of training epochs, run:
 `python3 trainTagger.py`

(note, it might be worth switching to RoBERTa using the `--bert_model` argument to the `bertTagger.py` file)


Once the models are trained, pick the best performing one and run the prediction command:

`python bertTagger.py --eval_data INPUT.csv --pred_output_file OUTPUT.pkl --do_eval --output_dir TRAINED_MODEL_DIR --txtCol TEXTCOL`

Here, the INPUT.csv contains one sentence per line, with the text in the TEXTCOL.
It will output a pandas pickled dataframe, where the columns of interest are:
- `toks`: the tokens/words in the sentence (note, some tokens are multiple BERT-tokens due to WordPiece tokenization, but the code re-aggregates them; for the non-aggregate WordPieces, see `toks_ids`)
- `predEventLabel`: the predicted realis tags; 1 if there is a realis event detected, 0 otherwise

Note, in order to train the model, please run `data/tsv/formatToDf.py` which will generate the train.csv, dev.csv and test.csv files.

