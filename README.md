# Search ranking comparison

A search ranking comparison between a standard BM25 search algorithm and a ranking methid that uses BM25 and re-ranks the results with BERT

## Getting started

If you follow the instructions in the bjarkivk/Eva repo you should be all set for using this repo for search renking comparison.

Now you have:

- Python installed
- Elasticsearch running on localhost:9200 with an index that has the Wikipedia data (see bjarki/Eva repo)

## Training a model

You will need a paragraphs file on the form of the `paragraphs.json` file from bjarkivk/Eva

The paragraphs file is specified as`paragraphsfile` in `kb_bert_training.py`

Then you can train a BERT model with (a GPU is needed for this):

`$ python kb_bert_training.py`

This creates positive and negative training exapmles for the fine-tuning of KB-BERT.

This saves a model to a file.

You can load the models you have trained when you search with `search.py` and `search_multiple.py`.

The model is defined in the beginning of `search.py` and `search_multiple.py`.

## Doing a single search

Load the desired model, it is defined as `model` in the beginning of the file.

Queries are on the form \[title/heading1/heading1.5\]. To do a single search:

`$ python search.py "[Thomasgymnasiet/Historia]"`

## Doing multiple searches

To do multiple searches with queries sampled from the json file specified as `PARAGRAPHS_FILE` in the `search_multiple.py` file, do:

`$ python search_multiple.py 10`

This will do 10 searches with randomly sampled queries from the `PARAGRAPHS_FILE`. Do not use the same `PARAGRAPHS_FILE` here as you used in training, otherwise there will be a leak from training data to the testing data.
