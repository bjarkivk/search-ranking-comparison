import pandas as pd  
import random
import json
import requests
import time
import sys
import os
import torch
from transformers import AutoTokenizer
import numpy as np
from operator import itemgetter
from SearchResult import SearchResult

####### Does multiple searches                   ###########
####### Compares the BM25 to the BERT re-ranker  ########


# Important constants that can be changed
MODEL_FILE = 'training_BERT/model_D_100k'                   # Them model in we use
PARAGRAPHS_FILE = 'training_BERT/paragraphs_chunked_4.json' # The test set we use
RESULTS_FILE = 'comparison_seed110_200.csv'                 # Name of the file where the results will appear in
RANK_P = 10                                                 # How many search results we evaluate, set to 10 because the re-ranker only re-ranks first 10 hits
random.seed(110)                                            # Random seed for sampling of the queries, can be commented out


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
model = torch.load(MODEL_FILE, map_location ='cpu') # When we only have cpu



### Functions ###

def get_searchArray(obj):
    a = []
    if obj["id1"] != "": 
        a.append(obj["id1"])
        if obj["id2"] != "": 
            a.append(obj["id2"])
            if obj["id3"] != "": 
                a.append(obj["id3"])
                if obj["id4"] != "": 
                    a.append(obj["id4"])
                    if obj["id5"] != "": 
                        a.append(obj["id5"])
    return a


def get_BERT_scores(query, top10_hits):
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

# Do one search query_tuple is (query with spaces, query as json object)
def one_search(query, json_obj):

    ### BM25 search ###

    searchArray = get_searchArray(json_obj)
    space = " "

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

    ground_truth_json = r.json()

    #############################################


    ### Calculate Normalized Discounted Cumulative Gain for the algorithms ###

    Result = SearchResult(query, query_level=len(searchArray))

    score_dict = get_ground_truth_score_dict(ground_truth_json)

    # IDCG calculations
    for i in range(RANK_P):
        # set relevance to zero if we dont have enough results in ground truth
        relevance_i_IDCG = 0.0
        if(i < len(ground_truth_json["hits"]["hits"])):
            relevance_i_IDCG = ground_truth_json["hits"]["hits"][i]["_score"]
        fraction_IDCG = relevance_i_IDCG/np.log2(i+1+1)

        Result.IDCG += fraction_IDCG 
        Result.IDCG_list.append(Result.IDCG)
        Result.ground_truth_relevance_list.append(relevance_i_IDCG)
        # print(relevance_i_IDCG, Result.IDCG, fraction_IDCG)

    # DCG_BM25 calculations
    for i in range(RANK_P):
        bm25_score = 0.0 # give the score zero as a placeholder if BM25 had less than RANK_P results
        if (i <= len(bm_25_json["hits"]["hits"]) - 1):
            bm25_id = bm_25_json["hits"]["hits"][i]["_id"] # get id of i-th search result in bm25
            bm25_score = score_dict.get(bm25_id, 0.0) # get score of i-th search result in bm25, if it doesn't exist it is zero
        relevance_i_DCG_BM25 = bm25_score
        fraction_DCG_BM25 = relevance_i_DCG_BM25/np.log2(i+1+1)

        Result.DCG_BM25 += fraction_DCG_BM25 
        Result.DCG_BM25_list.append(Result.DCG_BM25)
        Result.NDCG_BM25_list.append(Result.DCG_BM25_list[i]/Result.IDCG_list[i])
        Result.BM25_relevance_list.append(relevance_i_DCG_BM25)
        # print(relevance_i_DCG_BM25, Result.DCG_BM25, fraction_DCG_BM25)


    # DCG_re_rank calculations
    for i in range(RANK_P):
        re_rank_score = 0.0 # give the score zero as a placeholder if BM25 had less than RANK_P results
        if (i <= len(re_ranked_results_json["hits"]["hits"]) - 1):
            re_rank_id = re_ranked_results_json["hits"]["hits"][i]["_id"] # get id of i-th search result in re-ranked results
            re_rank_score = score_dict.get(re_rank_id, 0.0) # get score of i-th search result in bm25, if it doesn't exist it is zero
        relevance_i_DCG_re_rank = re_rank_score
        fraction_DCG_re_rank = relevance_i_DCG_re_rank/np.log2(i+1+1)

        Result.DCG_re_rank += fraction_DCG_re_rank
        Result.DCG_re_rank_list.append(Result.DCG_re_rank)
        Result.NDCG_re_rank_list.append(Result.DCG_re_rank_list[i]/Result.IDCG_list[i])
        Result.re_rank_relevance_list.append(relevance_i_DCG_re_rank)
        # print(relevance_i_DCG_re_rank, Result.DCG_re_rank, fraction_DCG_re_rank)


    return Result


    ##########################################################################



### Start of code ###

# Timekeeper
start = time.time()

# Read from json bulk upload file
paragraphsfile = open(PARAGRAPHS_FILE, 'r')
lines = paragraphsfile.readlines()

queries = {}

# Read every line and create test queries for every unique query
for index, line in enumerate(lines):
    line = line.replace('\n', '')
    if(line != '{"index":{}}'): # We discard the empty lines
        y = json.loads(line)
        q = y["id1"] + " " + y["id2"] + " " + y["id3"] + " " + y["id4"] + " " + y["id5"] # Create a query that is all ids concatinated together with space between
        query = q.strip() # Remove spaces at the end
        queries[query] = y

print("All queries", len(lines)/2)
print("Unique queries", len(queries))




# Sample from the test queries, quantity of searches defined in argument value
sampled_keys = random.sample(list(queries), int(sys.argv[1]))

# y = json.loads('{"id1": "Sm??var", "id2": "", "id3": "", "id4": "", "id5": "", "paragraph": "Sm??var, Zeugopterus norvegicus ??r en fisk i familjen piggvarar som ??r Europas minsta plattfisk. Den kallas ??ven sm??varv.[2]"}')
# sampled_list = [("Sm??var", y)]
# print("sampled_list",sampled_list)

results_list = []

for i, query in enumerate(sampled_keys):
    print(i+1, "/", len(sampled_keys), "Query:", query)
    res = one_search(query,queries[query])
    results_list.append(res)


bm25_wins = 0
re_rank_wins = 0
draws = 0

# initialize lists that will go into the pandas dataframe
queries_for_dataframe = []
query_level_for_daraframe = []
NDCG_BM25_for_dataframe = []
NDCG_re_rank_for_dataframe = []

for j in results_list:
    print("re-ranker advantage:",j.NDCG_re_rank_list[-1]-j.NDCG_BM25_list[-1], "bm25:", j.NDCG_BM25_list[-1], "re-ranker:", j.NDCG_re_rank_list[-1], "Query:", j.query)
    if j.NDCG_BM25_list[-1] > j.NDCG_re_rank_list[-1]:
        bm25_wins += 1
    elif j.NDCG_BM25_list[-1] < j.NDCG_re_rank_list[-1]:
        re_rank_wins += 1
    else:
        draws += 1
    queries_for_dataframe.append(j.query)
    query_level_for_daraframe.append(j.query_level)
    NDCG_BM25_for_dataframe.append(j.NDCG_BM25_list[-1])
    NDCG_re_rank_for_dataframe.append(j.NDCG_re_rank_list[-1])

    # print()
    # print("ground_truth_relevance_list")
    # [print(i) for i in j.ground_truth_relevance_list]

    # print()
    # print("BM25_relevance_list")
    # [print(i) for i in j.BM25_relevance_list]

    # print()
    # print("re_rank_relevance_list")
    # [print(i) for i in j.re_rank_relevance_list]

    # print()
    # print("IDCG_list")
    # [print(i) for i in j.IDCG_list]

    # print()
    # print("DCG_BM25_list")
    # [print(i) for i in j.DCG_BM25_list]

    # print()
    # print("DCG_re_rank_list")
    # [print(i) for i in j.DCG_re_rank_list]

    # print()
    # print("NDCG_BM25_list")
    # [print(i) for i in j.NDCG_BM25_list]

    # print()
    # print("NDCG_re_rank_list")
    # [print(i) for i in j.NDCG_re_rank_list]


print("bm25_wins", bm25_wins)
print("re_rank_wins", re_rank_wins)
print("draws", draws)

data = {'query': queries_for_dataframe, 'query_level': query_level_for_daraframe, 'bm25_NDCG_at_rank_10': NDCG_BM25_for_dataframe, 're_ranker_NDCG_at_rank_10': NDCG_re_rank_for_dataframe}  
df = pd.DataFrame(data) 
df.to_csv(RESULTS_FILE, sep='\t', encoding='utf-8')

end = time.time()
print("Time elapsed:", end - start)










