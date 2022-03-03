class SearchResult:
  def __init__(self, query):
    self.query = query

    self.IDCG = 0; self.IDCG_list = []  # Ideal Discounted Cumulative Gain, that is Discounted Cumulative Gain for the ground truth
    self.DCG_BM25 = 0; self.DCG_BM25_list = [] # Discounted Cumulative Gain for BM25
    self.DCG_re_rank = 0; self.DCG_re_rank_list = [] # Discounted Cumulative Gain for the Re-ranked results

    self.NDCG_BM25_list = [] # Normalized Discounted Cumulative Gain for BM25
    self.NDCG_re_rank_list = [] # Normalized Discounted Cumulative Gain for the Re-ranked results

    self.ground_truth_relevance_list = []
    self.BM25_relevance_list = []
    self.re_rank_relevance_list = []
