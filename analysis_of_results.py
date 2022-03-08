import matplotlib.pyplot as plt
import pandas as pd  

# Here we analyse the search result comparison done with search_multiple.py

df = pd.read_csv('comparison_seed12_5.csv', sep='\t') 

print(df)

# Add a column to the dataframe with the advantage of the re-ranker
advantage = []
for index, row in df.iterrows():
    advantage.append(row['re_ranker_NDCG_at_rank_10']-row['bm25_NDCG_at_rank_10'])
df['re_ranker_advantage'] = advantage
print(df)


# Plot
df.plot('bm25_NDCG_at_rank_10','re_ranker_NDCG_at_rank_10',kind='scatter', color='red')
plt.show()
