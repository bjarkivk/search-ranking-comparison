from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

### In this file we analyse the search result comparison done with search_multiple.py ###

# Load data from files
df1 = pd.read_csv('comparison_seed101_200.csv', sep='\t', index_col=0)
df2 = pd.read_csv('comparison_seed102_200.csv', sep='\t', index_col=0)
df3 = pd.read_csv('comparison_seed103_200.csv', sep='\t', index_col=0)
df4 = pd.read_csv('comparison_seed104_200.csv', sep='\t', index_col=0)
df5 = pd.read_csv('comparison_seed105_200.csv', sep='\t', index_col=0)
df6 = pd.read_csv('comparison_seed106_200.csv', sep='\t', index_col=0)
df7 = pd.read_csv('comparison_seed107_200.csv', sep='\t', index_col=0)
df8 = pd.read_csv('comparison_seed108_200.csv', sep='\t', index_col=0)
df9 = pd.read_csv('comparison_seed109_200.csv', sep='\t', index_col=0)
df10 = pd.read_csv('comparison_seed110_200.csv', sep='\t', index_col=0)



# print(df1)
# print(df2)


# Combine all dataframes into one
combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], ignore_index=True)
print(combined_df)

# Remove duplicate values in dataframe
no_dulicates_df = combined_df.drop_duplicates()
print("no_dulicates_df")
print(no_dulicates_df)

# Add a column to the dataframe with the advantage of the re-ranker
main_df = no_dulicates_df.copy(deep=True)
main_df['re_ranker_advantage'] = no_dulicates_df.apply( lambda row: row.re_ranker_NDCG_at_rank_10 - row.bm25_NDCG_at_rank_10, axis=1)
print(main_df)

# Best re-ranker efforts
print(main_df[main_df['re_ranker_advantage']>0.3])

# Find mean of the scores
bm25_NDCG_mean = main_df["bm25_NDCG_at_rank_10"].mean()
re_ranker_NDCG_mean = main_df["re_ranker_NDCG_at_rank_10"].mean()
print()
print("Mean values:")
print("bm25_NDCG_mean", bm25_NDCG_mean)
print("re_ranker_NDCG_mean", re_ranker_NDCG_mean)

# Find sum of the scores
bm25_NDCG_sum = main_df["bm25_NDCG_at_rank_10"].sum()
re_ranker_NDCG_sum = main_df["re_ranker_NDCG_at_rank_10"].sum()
print()
print("Sum values:")
print("bm25_NDCG_sum", bm25_NDCG_sum)
print("re_ranker_NDCG_sum", re_ranker_NDCG_sum)

# Find mean of the scores by query level
means_by_query_level = main_df.groupby('query_level').mean()
print("means_by_query_level")
print(means_by_query_level)

# Find count of samples by query level
count_by_query_level = main_df.groupby('query_level').size()
print("count_by_query_level")
print(count_by_query_level)

# Find which algorithm won the most often
re_ranker_wins = len(main_df[(main_df['re_ranker_advantage']>0)])
bm25_wins = len(main_df[(main_df['re_ranker_advantage']<0)])
draws = len(main_df[(main_df['re_ranker_advantage']==0)])
print("re_ranker_wins", re_ranker_wins)
print("bm25_wins", bm25_wins)
print("draws", draws)
print("total",re_ranker_wins + bm25_wins + draws )

# Dataframe with no samples where both algorithms return zero good results in top 10
no_zeros_df = main_df.copy(deep=True)
no_zeros_df = no_zeros_df[no_zeros_df.bm25_NDCG_at_rank_10 != 0.0]
print("no_zeros_df",no_zeros_df)

# # Dataframe with no draws
# no_draws_df = main_df.copy(deep=True)
# no_draws_df = no_draws_df[no_draws_df.bm25_NDCG_at_rank_10 != no_draws_df.re_ranker_NDCG_at_rank_10]
# print(no_draws_df)



# Plot the search data
main_df.plot('bm25_NDCG_at_rank_10','re_ranker_NDCG_at_rank_10',kind='scatter', color='red', alpha=0.3)
# no_zeros_df.plot('bm25_NDCG_at_rank_10','re_ranker_NDCG_at_rank_10',kind='scatter', color='blue')
# no_draws_df.plot('bm25_NDCG_at_rank_10','re_ranker_NDCG_at_rank_10',kind='scatter', color='red', alpha=0.3)

# Plot a line 
x = np.linspace(0, 1, 10)
# plt.plot(x, x, color='black', marker='o',linewidth=2, markersize=5)
# plt.plot(x, x, color='black')


plt.gca().set_aspect('equal', adjustable='box') # Set axis equal
plt.xlabel("BM25 NDCG at rank 10")
plt.ylabel("Re-ranker NDCG at rank 10")
plt.show()
