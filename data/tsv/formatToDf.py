# IPython log file
import pandas as pd
from IPython import embed
from glob import glob

for split in ["train", "test", "dev"]:
    allLines = {}
    for f in glob(split+"/*.tsv"):
        allLines[f] = [[t.split("\t") for t in l.split("\n")] for l in open(f).read().split("\n\n") if l.strip()]
    data = [[f,i]+list(zip(*l)) for f, lines in allLines.items() for i,l in enumerate(lines) if l]
    df = pd.DataFrame(data,columns=["filename","sentIx","toks","labels"])
    
    df.loc[df["labels"].isnull(),"labels"] = df.loc[df["labels"].isnull(),"toks"].apply(lambda x: tuple(["O" for i in x]))
    try:
        assert all(df["toks"].apply(len) == df["labels"].apply(len))
    except:
        embed()
    df[["toks","labels"]] = df[["toks","labels"]].apply(lambda x: x.apply(list))
    print(split, df.shape)
    df.to_csv(split+".csv")

