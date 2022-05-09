# NLP_sst2
Homework for sst2

## Dataset
Auto imported from Huggingface Datasets by `dataset` module.
Can also be grabbed from url: https://dl.fbaipublicfiles.com/glue/data/SST-2.zip

The `Stanford Sentiment Treebank (SST)` consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the `sentiment` of a given sentence. It uses the `two-way (positive 1 / negative 0)` class split, with only sentence-level labels. The following is a part of `train.tsv` from sst2 datasets.


| sentence                                                                                                                                              | label |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| hide new secretions from the parental units                                                                                                           | 0     |
| contains no wit , only labored gags                                                                                                                   | 0     |
| that loves its characters and communicates something rather beautiful about human nature                                                              | 1     |
| remains utterly satisfied to remain the same throughout                                                                                               | 0     |
| on the worst revenge-of-the-nerds clichés the filmmakers could dredge up                                                                             | 0     |
| that 's far too tragic to merit such superficial treatment                                                                                            | 0     |
| demonstrates that the director of such hollywood blockbusters as patriot games can still turn out a small , personal film with an emotional wallop .  | 1     |
| of saucy                                                                                                                                              | 1     |
| a depressed fifteen-year-old 's suicidal poetry                                                                                                       | 0     |
| are more deeply thought through than in most ` right-thinking ' films                                                                                 | 1     |
| goes to absurd lengths                                                                                                                                | 0     |
| for those moviegoers who complain that ` they do n't make movies like they used to anymore                                                            | 0     |
| the part where nothing 's happening ,                                                                                                                 | 0     |
| saw how bad this movie was                                                                                                                            | 0     |
| lend some dignity to a dumb story                                                                                                                     | 0     |
| the greatest musicians                                                                                                                                | 1     |
| cold movie                                                                                                                                            | 0     |
| with his usual intelligence and subtlety                                                                                                              | 1     |
| redundant concept                                                                                                                                     | 0     |


## Dependency
Create a virtual conda env
```
conda create -n glue python=3.8.1
conda activate glue
```

Install pyTorch
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
```

Then install requirements through `pip`
```
pip install -r requirements.txt
```

## Run Original sst2 benchmarks
Obtain original sst2 benchmarks using huggingface trainer
```
cd ori_glue
bash run.sh |& tee ori_sst2.log
```

## Run regularized dropout normailzed sst2 benchmarks
Obtain regularized dropout normailzed sst2 benchmarks using huggingface trainer
```
cd r_drop_glue
bash run.sh |& tee rdrop_sst2.log
```

## Simple Usage
In python shell

```
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="r_drop_glue/sst2_results")
# Or Running this
# classifier = pipeline("sentiment-analysis", model="ori_glue/sst2_results")

# Make prediction of the following sentences
classifier("how bad this movie was")
classifier("what a great idea")
```

## R-Drop changes
Change happened in [modeling_bert.py](r_drop_glue/transformers_rdrop/models/bert/modeling_bert.py)

In  `class BertForSequenceClassification` ,
