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
    if(os.path.exists("bm25_search_results.json")):
        os.remove("bm25_search_results.json")
    file = open("bm25_search_results.json", "a")
    file.write(obj)

def re_ranked_writeToFile(obj):
    if(os.path.exists("re_ranked_search_results.json")):
        os.remove("re_ranked_search_results.json")
    file = open("re_ranked_search_results.json", "a")
    file.write(obj)

def ground_truth_writeToFile(obj):
    if(os.path.exists("ground_truth_search_results.json")):
        os.remove("ground_truth_search_results.json")
    file = open("ground_truth_search_results.json", "a")
    file.write(obj)
    


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
    model = torch.load('training_BERT/model_D_100k',map_location ='cpu') # When we only have cpu

    
    queries = [ query for x in range(len(top10_hits)) ] # As many queries as there are many hits, could be less than 10
    paragraphs = [ i['_source']['paragraph'] for i in top10_hits ] # paragraphs of the top 10 hits
 
    model_inputs = tokenizer(queries, paragraphs, truncation='longest_first', padding='max_length', max_length=512, return_tensors="pt")


    # forward pass
    outputs = model(**model_inputs)
    na = outputs.logits.detach().numpy()
    bert_scores = na[:,1]

    # if the lowest BERT score is negative, we add the absolute value of it to every value
    lowest_score = min(bert_scores)
    if(lowest_score<0):
        final_scores = [x+abs(lowest_score) for x in bert_scores]
    else:
        final_scores = bert_scores

    return final_scores


def bert_re_ranking(top10, bert_scores):
    # print('bert_scores', bert_scores)
    # print('top10')
    # for x in top10:
    #     print(x['_score'])
    lowest_score_in_top10 = top10[-1]['_score']
    new_score = [ lowest_score_in_top10 + bert_scores[index] for index, value in enumerate(top10) ]
    # print('new_score', new_score)
    top10_with_new_scores =  []
    for index, x in enumerate(top10):
        item = x
        item['_score'] = new_score[index]
        top10_with_new_scores.append(item)
    
    # print('top10_with_added_score', top10_with_new_scores)


    new_ranking = sorted(top10_with_new_scores, key=itemgetter('_score'), reverse=True) 

    # print('new_ranking', new_ranking)

    return new_ranking

# Returns a dictionary with document id as key and score as value: {"6JpT_H4BGHonESiC523s": 3.0}
def get_ground_truth_score_dict(ground_truth_json):
    score_dict = {}
    for i in ground_truth_json["hits"]["hits"]:
        score_dict[i["_id"]] = i["_score"]
    return score_dict



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
bm_25_json = r.json()
bm25_writeToFile(bm25_results)

################


### BERT re-ranking ###

bm25_results_json = json.loads(bm25_results)
re_ranked_results_json = bm25_results_json
hits = bm25_results_json['hits']['hits']

# Get first 10 hits and re-rank them
top10 = hits[:10]
rest = hits[10:]

# if there are no search results, return empty list
all_hits = []
if len(top10) != 0:    
    bert_scores = get_BERT_scores(query, top10)
    re_ranked_results_top10 = bert_re_ranking(top10, bert_scores)
    all_hits = re_ranked_results_top10 + rest

re_ranked_results_json['hits']['hits'] = all_hits
re_ranked_results = json.dumps(re_ranked_results_json, ensure_ascii=False)
re_ranked_writeToFile(re_ranked_results)

########################


### Ground truth ranking ###

# Three Elasticsearch searches combined in one to fetch all paragraphs necessary for the ground truth of example [a/b/c/d/e]
payload = json.dumps({
  "size": 1000,
  "query": {
    "bool": {
      "should": [
        # First, fetch all paragraphs with exactly id=[a/b/c/d/e]
        {
          "constant_score": {
            "filter": {
              "bool": {
                "must": [
                  { "match_phrase": { "id1": searchArray[0] } },
                  { "match_phrase": { "id2": searchArray[1] if len(searchArray)>1 else "" } },
                  { "match_phrase": { "id3": searchArray[2] if len(searchArray)>2 else "" } },
                  { "match_phrase": { "id4": searchArray[3] if len(searchArray)>3 else "" } },
                  { "match_phrase": { "id5": searchArray[4] if len(searchArray)>4 else "" } },
                ],
              },
            },
            "boost": 3,
          },
        },

        # Second, fetch all paragraphs with id=[a/b/*/*/*]
        {
          "constant_score": {
            "filter": {
              "bool": {
                "must_not": {
                  "bool": {
                    "must": [
                      { "match_phrase": { "id1": searchArray[0] } },
                      { "match_phrase": { "id2": searchArray[1] if len(searchArray)>1 else "" } },
                      { "match_phrase": { "id3": searchArray[2] if len(searchArray)>2 else "" } },
                      { "match_phrase": { "id4": searchArray[3] if len(searchArray)>3 else "" } },
                      { "match_phrase": { "id5": searchArray[4] if len(searchArray)>4 else "" } },
                    ],
                  },
                },
                "must": [
                  { "match_phrase": { "id1": searchArray[0] } },
                  { "match_phrase": { "id2": searchArray[1] if len(searchArray)>1 else "" } },
                ],
              },
            },
            "boost": 2,
          },
        },

        # Third, fetch all paragraphs with id=[a/*/*/*/*]
        {
          "constant_score": {
            "filter": {
              "bool": {
                "must_not": {
                  "bool": {
                    "must": [
                      { "match_phrase": { "id1": searchArray[0] } },
                      { "match_phrase": { "id2": searchArray[1] if len(searchArray)>1 else "" } },
                    ],
                  },
                },

                "must": [{ "match_phrase": { "id1": searchArray[0] } }],
              },
            },
            "boost": 1,
          },
        },
      ],
    },
  },
})

headers = {"Content-Type": "application/json"}


r = requests.post('http://localhost:9200/paragraphs/_search/', data=payload, headers=headers)

ground_truth_results = r.text
ground_truth_json = r.json()
ground_truth_writeToFile(ground_truth_results)

#############################################


### Calculate Normalized Discounted Cumulative Gain for the algorithms ###

rank_p = 10
IDCG = 0; IDCG_list = []  # Ideal Discounted Cumulative Gain, that is Discounted Cumulative Gain for the ground truth
DCG_BM25 = 0; DCG_BM25_list = [] # Discounted Cumulative Gain for BM25
DCG_re_rank = 0; DCG_re_rank_list = [] # Discounted Cumulative Gain for the Re-ranked results

NDCG_BM25_list = [] # Normalized Discounted Cumulative Gain for BM25
NDCG_re_rank_list = [] # Normalized Discounted Cumulative Gain for the Re-ranked results

score_dict = get_ground_truth_score_dict(ground_truth_json)

print("IDCG calculations")
# IDCG calculations
for i in range(rank_p):
    # set relevance to zero if we dont have enough results in ground truth
    relevance_i_IDCG = 0.0
    if(i < len(ground_truth_json["hits"]["hits"])):
        relevance_i_IDCG = ground_truth_json["hits"]["hits"][i]["_score"]
    fraction_IDCG = relevance_i_IDCG/np.log2(i+1+1)
    IDCG += fraction_IDCG 
    IDCG_list.append(IDCG)
    print(relevance_i_IDCG, IDCG, fraction_IDCG)

print()
print("DCG_BM25 calculations")
# DCG_BM25 calculations
for i in range(rank_p):
    bm25_score = 0.0 # give the score zero as a placeholder if BM25 had less than rank_p results
    if (i <= len(bm_25_json["hits"]["hits"]) - 1):
        bm25_id = bm_25_json["hits"]["hits"][i]["_id"] # get id of i-th search result in bm25
        bm25_score = score_dict.get(bm25_id, 0.0) # get score of i-th search result in bm25, if it doesn't exist it is zero
    relevance_i_DCG_BM25 = bm25_score
    fraction_DCG_BM25 = relevance_i_DCG_BM25/np.log2(i+1+1)
    DCG_BM25 += fraction_DCG_BM25 
    DCG_BM25_list.append(DCG_BM25)
    NDCG_BM25_list.append(DCG_BM25_list[i]/IDCG_list[i])
    print(relevance_i_DCG_BM25, DCG_BM25, fraction_DCG_BM25)


print()
print("DCG_re_rank calculations")
# DCG_re_rank calculations
for i in range(rank_p):
    re_rank_score = 0.0 # give the score zero as a placeholder if BM25 had less than rank_p results
    if (i <= len(re_ranked_results_json["hits"]["hits"]) - 1):
        re_rank_id = re_ranked_results_json["hits"]["hits"][i]["_id"] # get id of i-th search result in re-ranked results
        re_rank_score = score_dict.get(re_rank_id, 0.0) # get score of i-th search result in bm25, if it doesn't exist it is zero
    relevance_i_DCG_re_rank = re_rank_score
    fraction_DCG_re_rank = relevance_i_DCG_re_rank/np.log2(i+1+1)
    DCG_re_rank += fraction_DCG_re_rank
    DCG_re_rank_list.append(DCG_re_rank)

    NDCG_re_rank_list.append(DCG_re_rank_list[i]/IDCG_list[i])
    print(relevance_i_DCG_re_rank, DCG_re_rank, fraction_DCG_re_rank)





print()
print("IDCG_list")
[print(i) for i in IDCG_list]

print()
print("DCG_BM25_list")
[print(i) for i in DCG_BM25_list]

print()
print("DCG_re_rank_list")
[print(i) for i in DCG_re_rank_list]

print()
print("NDCG_BM25_list")
[print(i) for i in NDCG_BM25_list]

print()
print("NDCG_re_rank_list")
[print(i) for i in NDCG_re_rank_list]


##########################################################################





