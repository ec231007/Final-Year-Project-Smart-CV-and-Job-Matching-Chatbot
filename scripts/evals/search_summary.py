import pandas as pd

# Paths
INPUT_CSV = "data/retrieval_metrics_balanced.csv"
OUTPUT_CSV = "data/retrieval_summary_report.csv"

# Load data
df = pd.read_csv(INPUT_CSV)

# -------------------------------
# Step 1: Select BEST config per file
# (based on strongest ranking signal)
# -------------------------------

# Sort by key performance metrics (priority order)
df_sorted = df.sort_values(
    by=["consensus_score", "mrr", "hit_at_1"],
    ascending=False
)

# Keep only the best config per file
best_per_file = df_sorted.drop_duplicates(subset=["filename"], keep="first")

# -------------------------------
# Step 2: Aggregate by category
# -------------------------------

summary = best_per_file.groupby("category").agg(
    total_queries=("filename", "count"),
    
    # Core performance
    success_rate=("search_success", "mean"),
    hit_at_1=("hit_at_1", "mean"),
    hit_at_5=("hit_at_5", "mean"),
    hit_at_10=("hit_at_10", "mean"),
    mrr=("mrr", "mean"),
    
    # Retrieval quality
    in_category_ratio=("in_cat_ratio", "mean"),
    
    # Scoring signals
    avg_bge_score=("avg_bge_score", "mean"),
    avg_llama_score=("llama_top_score", "mean"),
    avg_consensus_score=("consensus_score", "mean"),
    
    # Efficiency
    avg_retrieved=("count_retrieved", "mean")
).reset_index()

# -------------------------------
# Step 3: Clean formatting
# -------------------------------

summary = summary.round({
    "success_rate": 3,
    "hit_at_1": 3,
    "hit_at_5": 3,
    "hit_at_10": 3,
    "mrr": 3,
    "in_category_ratio": 3,
    "avg_bge_score": 4,
    "avg_llama_score": 4,
    "avg_consensus_score": 4,
    "avg_retrieved": 2
})

# Optional: sort by performance
summary = summary.sort_values(by="hit_at_1", ascending=False)

# Save
summary.to_csv(OUTPUT_CSV, index=False)

print("Saved to:", OUTPUT_CSV)
print(summary)