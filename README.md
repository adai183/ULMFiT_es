# ULMFit ES
This repository contains implementations of [ULMFit](https://arxiv.org/abs/1801.06146) by Jeremy Howard and Sebastian Ruder applied to NLP tasks for the Spanish Language.

Francisco Ingham [project](https://github.com/fpingham/SpanishULMFit)  for *Classification of Spanish Tweets as of 2017 in the GeneralTASS Dataset* was used as a starting point for hyperparameter tuning.

A F1(macro) score of 0.7026 is achieved(see [notebook](https://github.com/adai183/ULMFiT_es/blob/master/election_tweets.ipynb)). The fine-tuning and training of the classifier should not take longer than a couple of minutes on a machine suitable for deep-learning with a very basic GPU.



## Setup

```
git clone https://github.com/adai183/ULMFiT_es.git
cd ULMFiT_es
conda env create -f environment.yml
conda activate fastai_v07
jupyter-notebook

```
