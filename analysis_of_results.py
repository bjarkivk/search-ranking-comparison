import matplotlib.pyplot as plt
import pandas as pd  

### In this file we analyse the search result comparison done with search_multiple.py ###

# Load data from files
df1 = pd.read_csv('comparison_seed101_200.csv', sep='\t', index_col=0)
df2 = pd.read_csv('comparison_seed102_200.csv', sep='\t', index_col=0)


print(df1)
print(df2)


# Combine all dataframes into one
combined_df = pd.concat([df1, df2], ignore_index=True)
print(combined_df)

# Remove duplicate values in dataframe
no_dulicates_df = combined_df.drop_duplicates()
print("no_dulicates_df")
print(no_dulicates_df)

# Add a column to the dataframe with the advantage of the re-ranker
main_df = no_dulicates_df.copy(deep=True)
main_df['re_ranker_advantage'] = no_dulicates_df.apply( lambda row: row.re_ranker_NDCG_at_rank_10 - row.bm25_NDCG_at_rank_10, axis=1)
print(main_df)


# Find mean of the scores
bm25_NDCG_mean = main_df["bm25_NDCG_at_rank_10"].mean()
re_ranker_NDCG_mean = main_df["re_ranker_NDCG_at_rank_10"].mean()
print("bm25_NDCG_mean", bm25_NDCG_mean)
print("re_ranker_NDCG_mean", re_ranker_NDCG_mean)

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


# Plot
main_df.plot('bm25_NDCG_at_rank_10','re_ranker_NDCG_at_rank_10',kind='scatter', color='red')
plt.show()
