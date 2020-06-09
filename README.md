# Small BART for summarization of texts in Russian - project report
*Project for NLP course by Huawei, spring 2020.*

*Done by Ignatov Dmitry. Date of report 30.05.2020*

*An article on the work done is [here](https://www.dropbox.com/s/lc2vug0s5tjluaa/Small_BART_for_the_problem_of_summarizing_the_Russian_language.pdf?dl=0).*

### Introduction

The [BART](https://arxiv.org/pdf/1910.13461.pdf) architecture has shown excellent results in a wide range of tasksand has 
become the most advanced for the summarization task.  To achievesuch results, a large 
model was used that required a huge amount of computingresources. In this study, a 
reduced copy of BART will be considered. Considerits capabilities for the problem of 
summarizing the Russian text.

### Tutorials
- [Create Russian Wikipedia dataset](./examples/ruWiki.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IgnatovD/ruBart/blob/master/examples/ruWiki.ipynb)
- [Train the tokenizer from scratch](./examples/train_tokenizer.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IgnatovD/ruBart/blob/master/examples/train_tokenizer.ipynb)
- [Pre-training a small BART](./examples/TrainerMLM.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IgnatovD/ruBart/blob/master/examples/TrainerMLM.ipynb)
- [Fine-tune a small BART for the task of summarizing texts](./examples/FineTune.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IgnatovD/ruBart/blob/master/examples/FineTune.ipynb)