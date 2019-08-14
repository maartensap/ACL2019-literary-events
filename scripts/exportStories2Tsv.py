#!/usr/bin/env python3
import os
import pandas as pd
import argparse
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from IPython import embed

def main(args):
  tqdm.pandas(ascii=True)
  df = pd.read_csv(args.input_file)
  df["tokens"] = df[args.text_column].progress_apply(word_tokenize)
  
  assert df.AssignmentId.nunique() == len(df)
  df = df.set_index("AssignmentId")

  if not os.path.exists(args.output_dir):
    print("{} doesn't exist, creating directory structure".format(args.output_dir))
    os.makedirs(args.output_dir)
  
  for aid, toks in df["tokens"].iteritems():
    outf = os.path.join(args.output_dir,aid+".tsv")
    tokDf = pd.DataFrame(toks,columns=["tok"])
    tokDf["event"] = "O"
    tokDf.to_csv(outf,sep="\t",header=False,index=False)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file")
  parser.add_argument("--text_column",default="story")
  parser.add_argument("--output_dir")
  args = parser.parse_args()
  if not args.input_file or not args.output_dir:
    parser.print_help()
    exit()
    
  main(args)
