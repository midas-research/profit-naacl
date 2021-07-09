# Quantitative Day Trading from Natural Language using Reinforcement Learning

This codebase contains the python scripts for developing the model called PROFIT, proposed in the [paper](https://aclanthology.org/2021.naacl-main.316.pdf) titled "Quantitative Day Trading from Natural Language using Reinforcement Learning".

Published at NAACL - 2021 Annual Conference of the North American Chapter of the Association for Computational Linguistics

## Environment & Installation Steps
Create an environment and install the following dependencies
```python
pip install -r requirements.txt
```

## Contents and Description

configs_stock.py -> This file contains all the configuration parameters for the stock market environment.

ddpg.py -> This file contains an implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.

env.py -> This file contains the code for the stock market environment.

evaluator.py -> This file contains the implementation for evaluating the performance of PROFIT across different metrics.

main.py -> This file starts the training of the model.

memory.py -> This file implements the memory module for the DDPG algorithm.

model.py -> This file contains a heirarchical time-aware attention and LSTM based Actor & Critic model.

random_process.py -> This file contains implementation of some methods for random sampling for the DDPG algorithm.

util.py -> This file contains implementation of utility functions.

## Data and Preprocessing

Find the US S&P 500 dataset [here](https://github.com/yumoxu/stocknet-dataset), and the China & Hong Kong dataset [here](https://pan.baidu.com/s/1mhCLJJi).

To encode the texts, we use the 768-dimensional embedding obtained per news item or tweet by averaging the token-level outputs from the final layer of BERT. However, PROFIT is compatible with any and all 1-D text embeddings.
To extract the timestamp input for time-aware LSTM, we obtain the time interval (in minutes) between the release of two consecutive texts and compute its inverse.
Kindly refer the [paper](https://aclanthology.org/2021.naacl-main.316.pdf) for further pre-processing details and the model training setup.

Prepare two sepearte .pkl files, one for training and one for testing data, containing data processed as follows. Each data point should comprise values corresponding to the following keys for the set of stocks during the lookback window:

'dates' -> Dates (list) corresponding to the days in the lag window

'date_target' -> The date for the target trading day (trading day next to the last day in the lookback)

'date_last' -> The date for the last day in the lookback

'embedding' -> Data embeddings with dimensions as follows: `[number of stocks, length of lookback, number of financial texts per day, embedding size of financial text]`

'length_data' -> Denotes the number of texts for each day in the lag per stock, dimensions of tensor: `[number of stocks, length of lookback]`

'time_features' -> Inverse of time gap between release of two consecutive texts, dimensions of tensor: `[number of stocks, length of lookback, number of texts per day, 1]`

'adj_close_last' -> A list of adjusted closing prices for the set of stocks on the day before the target trading day

'adj_close_target' -> A list of adjusted closing prices for the set of stocks on the target trading day

## Training
Execute the following command in the same environment to train PROFIT:
```bash
python main.py --model profit --train_iter 1000
```

## Cite

If our work was helpful for your research, please don't forget to cite our work.

    @inproceedings{sawhney-etal-2021-quantitative,
        title = "Quantitative Day Trading from Natural Language using Reinforcement Learning",
        author = "Sawhney, Ramit  and
          Wadhwa, Arnav  and
          Agarwal, Shivam  and
          Shah, Rajiv Ratn",
        booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
        month = jun,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.naacl-main.316",
        doi = "10.18653/v1/2021.naacl-main.316",
        pages = "4018--4030",
        abstract = "It is challenging to design profitable and practical trading strategies, as stock price movements are highly stochastic, and the market is heavily influenced by chaotic data across sources like news and social media. Existing NLP approaches largely treat stock prediction as a classification or regression problem and are not optimized to make profitable investment decisions. Further, they do not model the temporal dynamics of large volumes of diversely influential text to which the market responds quickly. Building on these shortcomings, we propose a deep reinforcement learning approach that makes time-aware decisions to trade stocks while optimizing profit using textual data. Our method outperforms state-of-the-art in terms of risk-adjusted returns in trading simulations on two benchmarks: Tweets (English) and financial news (Chinese) pertaining to two major indexes and four global stock markets. Through extensive experiments and studies, we build the case for our method as a tool for quantitative trading.",
    }