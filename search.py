import json
import requests
import sys
import os
import torch
from transformers import AutoTokenizer
import numpy as np
from operator import itemgetter



### Functions ###

def bm25_writeToFile(obj):
    if(os.path.exists("BM25_SearchResults.json")):
        os.remove("BM25_SearchResults.json")
    bm25_search_results_file = open("BM25_SearchResults.json", "a")
    bm25_search_results_file.write(obj)

def re_ranked_writeToFile(obj):
    if(os.path.exists("Re_ranked_SearchResults.json")):
        os.remove("Re_ranked_SearchResults.json")
    bm25_bert_search_results_file = open("Re_ranked_SearchResults.json", "a")
    bm25_bert_search_results_file.write(obj)
    


def seperate(arg):
    firstChar = arg[0]
    lastChar = arg[len(arg) - 1]
    if (firstChar == "[" and lastChar == "]"):
        str = arg[1:-1]
        array = str.split("/")
        return array
    else:
        raise Exception("Search term not on correct format")


def get_BERT_scores(query, top10_hits):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
    model = torch.load('training_BERT/model_B_10k',map_location ='cpu') # When we only have cpu

    
    queries = [ query for x in range(10) ] # ten times the same query
    paragraphs = [ i['_source']['paragraph'] for i in top10_hits ] # paragraphs of the top 10 hits
 
    model_inputs = tokenizer(queries, paragraphs, truncation='longest_first', padding='max_length', max_length=512, return_tensors="pt")


    # forward pass
    outputs = model(**model_inputs)
    na = outputs.logits.detach().numpy()
    bert_scores = na[:,1]
    return bert_scores


def bert_re_ranking(top10, bert_scores):
    print('bert_scores', bert_scores)
    print('top10')
    for x in top10:
        print(x['_score'])
    combined_score = [ value['_score']+ bert_scores[index] for index, value in enumerate(top10) ]
    print('combined_score', combined_score)
    top10_with_new_scores =  []
    for index, x in enumerate(top10):
        item = x
        item['_score'] = combined_score[index]
        top10_with_new_scores.append(item)
    
    print('top10_with_added_score', top10_with_new_scores)


    new_ranking = sorted(top10_with_new_scores, key=itemgetter('_score'), reverse=True) 

    print('new_ranking', new_ranking)

    return new_ranking

###################


### BM25 search ###

arg = sys.argv[1]
searchArray = seperate(arg)
space = " "
query = space.join(searchArray)




payload = json.dumps({
  "size": 1000,
  "query": {
    "match": {
      "paragraph": query,
    },
  },
})
headers = {"Content-Type": "application/json"}


r = requests.post('http://localhost:9200/paragraphs/_search/', data=payload, headers=headers)

bm25_results = r.text
bm25_writeToFile(bm25_results)

################


### BERT re-ranking ###

bm25_results_json = json.loads(bm25_results)
re_ranked_results_json = bm25_results_json
hits = bm25_results_json['hits']['hits']

# Get first 10 hits and re-rank them
top10 = hits[:10]
rest = hits[10:]

bert_scores = get_BERT_scores(query, top10)
re_ranked_results = bert_re_ranking(top10, bert_scores)


all_hits = re_ranked_results + rest
re_ranked_results_json['hits']['hits']=all_hits
re_ranked_writeToFile(json.dumps(re_ranked_results_json, ensure_ascii=False))





