import os
from IPython import embed

for e in [1, 2, 3, 4, 5]:
    scoreDict = {}
    for i in range(3):
        print(e,i)
        cmd = f"python bertTagger.py --eval_data ../data/tsv/dev.csv --do_eval --output_dir realisTagger_e{e}_r{i} --do_train --train_data ../data/tsv/train.csv --num_train_epochs {e} --seed {i}"
        print(cmd)
        os.system(cmd)
        # f1=`tail -n 1 realisTagger_e${e}_r${i}/log.txt | grep -Po "'f1': 0\.\d{0,3}" | sed -r "s/'f1[ \':]+//g"`

        try:
            scores = [l.strip() for l in open(f"realisTagger_e{e}_r{i}/log.txt")][-1].replace("INFO:__main__:","")
            scores = eval(scores)
            f1 = scores["f1"]
            scoreDict[i] = f1
            print(scoreDict)
            
            maxScore = max(scoreDict.values())
            for j, f in scoreDict.items():
                if f < maxScore:
                    print(f"F1 score lower than max score, deleting run: realisTagger_e{e}_r{j}/")
                    os.system(f"rm -r realisTagger_e{e}_r{j}/")
                    del scoreDict[j]
        except:
            pass
        
        # echo "${e},${i},${f1}" # >> realisTaggerF1dev.log

