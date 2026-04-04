import pandas as pd

# File paths
INPUT_CSV = "data/extraction_metrics_report.csv"
OUTPUT_CSV = "data/summary_metrics_report.csv"

# Load data
df = pd.read_csv(INPUT_CSV)

# Create summary table grouped by category
summary = df.groupby("category").agg(
    total_files=("filename", "count"),
    avg_ner_tag_count=("ner_tag_count", "mean"),
    avg_intent_fill_rate=("intent_fill_rate", "mean"),
    avg_semantic_similarity=("semantic_similarity", "mean"),
    intent_models_used=("intent_model", lambda x: x.nunique()),
    boost_models_used=("boost_model", lambda x: x.nunique())
).reset_index()

# Optional: round numeric values for report readability
summary["avg_ner_tag_count"] = summary["avg_ner_tag_count"].round(2)
summary["avg_intent_fill_rate"] = summary["avg_intent_fill_rate"].round(3)
summary["avg_semantic_similarity"] = summary["avg_semantic_similarity"].round(4)

# Save to CSV
summary.to_csv(OUTPUT_CSV, index=False)

print("Summary table saved to:", OUTPUT_CSV)
print(summary)