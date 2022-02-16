import json
from random import randrange





queries = []
paragraphs = []
labels = []


# Real Dataset, read from paragraphs.json bulk upload file
# Open the list of articles to read
paragraphsfile = open('../training_BERT/paragraphs_test.json', 'r')
lines = paragraphsfile.readlines()

# Create positive examples (label = 1)
for index, line in enumerate(lines):
    line = line.replace('\n', '')
    if(line != '{"index":{}}'): # We discard the empty lines
        y = json.loads(line)
        q = y["id1"] + " " + y["id2"] + " " + y["id3"] + " " + y["id4"] + " " + y["id5"] # Create a query that is all ids concatinated together with space between
        query = q.strip() # Remove spaces at the end
        queries.append(query)
        paragraphs.append(y["paragraph"])
        labels.append(1) # these are all positive examples, that is these paragraphs are relevant to the query

pos_count = len(queries)

print("Positive")
print(len(queries), queries)
print(len(paragraphs), paragraphs)
print(len(labels), labels)
print()

# Create as many negative examples as positive (negative: label = 0)
neg_count = pos_count
neg_queries = []
neg_paragraphs = []
neg_labels = []
for i in range(neg_count):
    random_index = randrange(neg_count)
    # Find a random paragraph that is paired with another query than its correct query
    while (queries[random_index].partition(' ')[0] == queries[i].partition(' ')[0]): # find new pair if first word of query is the same
        random_index = randrange(neg_count)
    neg_queries.append(queries[i])
    neg_paragraphs.append(paragraphs[random_index])
    neg_labels.append(0) # these are all negative examples, that is these paragraphs are not relevant to the query


print("Negative")
print(len(neg_queries), neg_queries)
print(len(neg_paragraphs), neg_paragraphs)
print(len(neg_labels), neg_labels)
print()

queries.extend(neg_queries)
paragraphs.extend(neg_paragraphs)
labels.extend(neg_labels)

print("Both")
print(len(queries), queries)
print(len(paragraphs), paragraphs)
print(len(labels), labels)
print()