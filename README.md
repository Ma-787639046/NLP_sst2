# NLP_sst2
Homework for sst2

## Dataset
Auto imported from Huggingface Datasets by `dataset` module.
Can also be grabbed from url: https://dl.fbaipublicfiles.com/glue/data/SST-2.zip

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

## R-Drop changes
Change happened in [modeling_bert.py](r_drop_glue/transformers_rdrop/models/bert/modeling_bert.py)

In  `class BertForSequenceClassification` ,
