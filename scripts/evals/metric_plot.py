import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Create output directory
output_dir = "fyp_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Load Data
try:
    df_ret = pd.read_csv(PROJECT_ROOT / 'data' / 'retrieval_metrics_balanced.csv')
    df_ext = pd.read_csv(PROJECT_ROOT / 'data' / 'extraction_metrics_report.csv')
    print("Files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure CSV files are in the current directory.")
    exit()

sns.set_theme(style="whitegrid")
palette = "viridis"

# ---------------------------------------------------------
# SECTION 1: EXTRACTION QUALITY (extraction_metrics_report.csv)
# ---------------------------------------------------------

# 1. NER Density by Industry
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='ner_tag_count', data=df_ext)
plt.xticks(rotation=45)
plt.title('Graph 1: Information Density (NER Tag Count) per Job Category')
plt.savefig(f"{output_dir}/1_ner_density_cat.png", bbox_inches='tight')

# 2. Intent Fill Rate Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df_ext['intent_fill_rate'], bins=20, kde=True, color='teal')
plt.title('Graph 2: Global Intent Completion (Data Extraction Completeness)')
plt.savefig(f"{output_dir}/2_intent_fill_dist.png")

# 3. Extraction Fidelity (Semantic Similarity)
plt.figure(figsize=(10, 5))
sns.kdeplot(df_ext['semantic_similarity'], fill=True, color='purple')
plt.title('Graph 3: Semantic Preservation (Similarity of Extracted vs Raw Text)')
plt.savefig(f"{output_dir}/3_extraction_fidelity.png")

# 4. Intent Fill Rate per Category
plt.figure(figsize=(12, 6))
sns.barplot(x='category', y='intent_fill_rate', data=df_ext, estimator=np.mean)
plt.xticks(rotation=45)
plt.title('Graph 4: Average Information Capture Rate by Category')
plt.savefig(f"{output_dir}/4_fill_rate_cat.png", bbox_inches='tight')

# ---------------------------------------------------------
# SECTION 2: RETRIEVAL ENGINE BENCHMARKING (retrieval_metrics_balanced.csv)
# ---------------------------------------------------------

# 5. Global Accuracy (Hit@K) by Configuration
hit_k = df_ret.groupby('config')[['hit_at_1', 'hit_at_5', 'hit_at_10']].mean().reset_index()
hit_k_melted = hit_k.melt(id_vars='config', var_name='Metric', value_name='Rate')
plt.figure(figsize=(12, 6))
sns.barplot(x='config', y='Rate', hue='Metric', data=hit_k_melted)
plt.title('Graph 5: Retrieval Accuracy (Hit@K) Comparison across Pipeline Toggles')
plt.savefig(f"{output_dir}/5_hit_k_configs.png")

# 6. Ranking Performance (MRR) by Config
plt.figure(figsize=(10, 6))
sns.barplot(x='config', y='mrr', data=df_ret, palette='magma')
plt.title('Graph 6: Mean Reciprocal Rank (MRR) - How high did the correct job rank?')
plt.savefig(f"{output_dir}/6_mrr_configs.png")

# 7. Heatmap: Performance Matrix (Category vs Config)
pivot_mrr = df_ret.pivot_table(index='category', columns='config', values='mrr', aggfunc='mean')
plt.figure(figsize=(14, 8))
sns.heatmap(pivot_mrr, annot=True, cmap='YlGnBu')
plt.title('Graph 7: MRR Heatmap (Category vs Pipeline Configuration)')
plt.savefig(f"{output_dir}/7_mrr_heatmap.png", bbox_inches='tight')

# 8. Precision (In-Category Ratio)
plt.figure(figsize=(10, 6))
sns.barplot(x='config', y='in_cat_ratio', data=df_ret)
plt.title('Graph 8: Precision - Ratio of Relevant Jobs in Top 10 Results')
plt.savefig(f"{output_dir}/8_precision_in_cat.png")

# 9. Search Success Rate (Did we find anything?)
success = df_ret.groupby(['config', 'category'])['search_success'].mean().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='category', y='search_success', hue='config', data=success, marker='o')
plt.xticks(rotation=45)
plt.title('Graph 9: System Reliability (Success Rate) per Category')
plt.savefig(f"{output_dir}/9_search_reliability.png", bbox_inches='tight')

# ---------------------------------------------------------
# SECTION 3: ENSEMBLE SCORING ANALYSIS
# ---------------------------------------------------------

# 10. Specialist vs. Generalist (Llama vs BGE)
plt.figure(figsize=(10, 6))
sns.kdeplot(df_ret['avg_bge_score'], label='BGE (Generalist)', fill=True)
sns.kdeplot(df_ret['llama_top_score'], label='Llama (Specialist)', fill=True)
plt.title('Graph 10: Score Calibration - How the Specialist Judge compares to the Search Engine')
plt.legend()
plt.savefig(f"{output_dir}/10_score_calibration.png")

# 11. Consensus Score Correlation
plt.figure(figsize=(8, 8))
sns.scatterplot(x='avg_bge_score', y='llama_top_score', hue='config', data=df_ret, alpha=0.5)
plt.title('Graph 11: Correlation between Semantic Match and LLM Judgment')
plt.savefig(f"{output_dir}/11_bge_vs_llama_scatter.png")

# 12. Average Consensus Score by Category
plt.figure(figsize=(12, 6))
sns.barplot(x='category', y='consensus_score', data=df_ret)
plt.xticks(rotation=45)
plt.title('Graph 12: Average Profile Match Strength (Consensus) by Industry')
plt.savefig(f"{output_dir}/12_match_strength_cat.png", bbox_inches='tight')

# ---------------------------------------------------------
# SECTION 4: IMPACT OF INDIVIDUAL FEATURES (Ablation Study)
# ---------------------------------------------------------

# 13. The "LLM Lift" (Comparison of configs with LLM:True vs LLM:False)
df_ret['LLM_Enabled'] = df_ret['config'].apply(lambda x: 'LLM:True' in x)
plt.figure(figsize=(8, 6))
sns.boxplot(x='LLM_Enabled', y='mrr', data=df_ret)
plt.title('Graph 13: Ablation Study - Impact of LLM Specialist on Ranking Accuracy')
plt.savefig(f"{output_dir}/13_llm_impact.png")

# 14. The "NER Lift" (Comparison of configs with NER:True vs NER:False)
df_ret['NER_Enabled'] = df_ret['config'].apply(lambda x: 'NER:True' in x)
plt.figure(figsize=(8, 6))
sns.boxplot(x='NER_Enabled', y='mrr', data=df_ret)
plt.title('Graph 14: Ablation Study - Impact of Structured Metadata (NER) on MRR')
plt.savefig(f"{output_dir}/14_ner_impact.png")

# ---------------------------------------------------------
# SECTION 5: CORRELATION (Extraction vs. Retrieval)
# ---------------------------------------------------------

# Merge datasets on filename/category to see if better extraction = better retrieval
merged = pd.merge(df_ret, df_ext, on=['filename', 'category'])

# 15. Metadata Richness vs Success
plt.figure(figsize=(10, 6))
sns.regplot(x='ner_tag_count', y='mrr', data=merged, scatter_kws={'alpha':0.1}, line_kws={'color':'red'})
plt.title('Graph 15: Does extracting more data lead to better ranking results?')
plt.savefig(f"{output_dir}/15_ner_vs_mrr.png")

# 16. Fill Rate vs. Success
plt.figure(figsize=(10, 6))
sns.boxplot(x=pd.cut(merged['intent_fill_rate'], bins=3), y='mrr', data=merged)
plt.title('Graph 16: Impact of Intent Completeness on Final MRR')
plt.savefig(f"{output_dir}/16_fill_vs_mrr.png")

# ---------------------------------------------------------
# SECTION 6: CATEGORY STRESS TEST
# ---------------------------------------------------------

# 17. Hardest Categories (Lowest Hit@1)
worst_cats = df_ret.groupby('category')['hit_at_1'].mean().sort_values().head(5)
plt.figure(figsize=(10, 5))
worst_cats.plot(kind='barh', color='salmon')
plt.title('Graph 17: "The Hard Five" - Categories with lowest Hit@1 rate')
plt.savefig(f"{output_dir}/17_worst_categories.png")

# 18. Best Categories (Highest Hit@1)
best_cats = df_ret.groupby('category')['hit_at_1'].mean().sort_values().tail(5)
plt.figure(figsize=(10, 5))
best_cats.plot(kind='barh', color='skyblue')
plt.title('Graph 18: "The Top Five" - Most compatible categories for retrieval')
plt.savefig(f"{output_dir}/18_best_categories.png")

# 19. Count Retrieved (Search Recall)
plt.figure(figsize=(12, 6))
sns.violinplot(x='category', y='count_retrieved', data=df_ret)
plt.xticks(rotation=45)
plt.title('Graph 19: Retrieval Density - Average number of job matches found')
plt.savefig(f"{output_dir}/19_retrieval_density.png", bbox_inches='tight')

# 20. Gain Analysis (Final Improvement)
# Delta between worst config (False/False) and best (True/True)
base = df_ret[df_ret['config'] == 'NER:False_LLM:False'].groupby('category')['mrr'].mean()
best = df_ret[df_ret['config'] == 'NER:True_LLM:True'].groupby('category')['mrr'].mean()
improvement = (best - base).sort_values()
plt.figure(figsize=(12, 6))
improvement.plot(kind='bar', color='forestgreen')
plt.xticks(rotation=45)
plt.title('Graph 20: Performance Lift - Net MRR Gain of Full Pipeline over Baseline')
plt.savefig(f"{output_dir}/20_pipeline_gain.png", bbox_inches='tight')

# Pre-processing: Create a combined dataframe for cross-analysis
merged = pd.merge(df_ret, df_ext, on=['filename', 'category'])

sns.set_theme(style="white")
# ---------------------------------------------------------
# SECTION 1: SYSTEM RELIABILITY & FAILURE ANALYSIS
# ---------------------------------------------------------

# 1. Failure Rate by Category (1 - search_success)
plt.figure(figsize=(12, 6))
fail_rate = (1 - df_ret.groupby('category')['search_success'].mean()).sort_values()
fail_rate.plot(kind='bar', color='firebrick')
plt.title('Graph 1: Retrieval Failure Rate by Category (System Blind Spots)')
plt.ylabel('Rate of Zero Results Found')
plt.savefig(f"{output_dir}/1_failure_rate.png", bbox_inches='tight')

# 2. MRR Stability (Coefficient of Variation)
# High variance means the system is "hit or miss" for that category.
plt.figure(figsize=(12, 6))
mrr_stats = df_ret.groupby('category')['mrr'].agg(['mean', 'std'])
mrr_stats['cv'] = mrr_stats['std'] / mrr_stats['mean']
mrr_stats['cv'].sort_values().plot(kind='bar', color='orchid')
plt.title('Graph 2: Ranking Volatility (Which categories are inconsistent?)')
plt.savefig(f"{output_dir}/2_mrr_volatility.png", bbox_inches='tight')

# 3. Success vs NER Density (Binned)
# We bin the NER counts to avoid scatter plot clutter.
merged['ner_bin'] = pd.qcut(merged['ner_tag_count'], q=4, labels=['Low', 'Medium', 'High', 'Expert'])
plt.figure(figsize=(10, 6))
sns.pointplot(x='ner_bin', y='search_success', hue='config', data=merged)
plt.title('Graph 3: Search Success Probability vs. Extraction Density')
plt.savefig(f"{output_dir}/3_success_prob_bins.png")

# ---------------------------------------------------------
# SECTION 2: THE "JUDGE" DISAGREEMENT (BGE vs LLM)
# ---------------------------------------------------------

# 4. The "Skepticism" Metric (BGE Score - Llama Score)
# If Llama is much lower than BGE, it means the LLM is acting as a filter.
df_ret['score_diff'] = df_ret['avg_bge_score'] - df_ret['llama_top_score']
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='score_diff', data=df_ret)
plt.xticks(rotation=45)
plt.title('Graph 4: Specialist Skepticism (BGE vs Llama Score Gap per Industry)')
plt.savefig(f"{output_dir}/4_judge_disagreement.png", bbox_inches='tight')

# 5. Consensus Confidence Distribution (Success vs Fail)
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_ret, x='consensus_score', hue='hit_at_1', fill=True, common_norm=False)
plt.title('Graph 5: Consensus Score Confidence for Hits vs Misses')
plt.savefig(f"{output_dir}/5_consensus_confidence.png")

# 6. LLM "Decisiveness" (Standard Deviation of Llama Scores per category)
plt.figure(figsize=(12, 6))
df_ret.groupby('category')['llama_top_score'].std().sort_values().plot(kind='bar', color='teal')
plt.title('Graph 6: LLM Decisiveness (Which industries have clear vs ambiguous candidates?)')
plt.savefig(f"{output_dir}/6_llm_decisiveness.png", bbox_inches='tight')

# ---------------------------------------------------------
# SECTION 3: ABLATION & CONFIG DOMINANCE
# ---------------------------------------------------------

# 7. Config Win-Rate (Which config produced the highest MRR for each category?)
best_configs = pivot_mrr = df_ret.groupby(['category', 'config'])['mrr'].mean().unstack()
win_counts = best_configs.idxmax(axis=1).value_counts()
plt.figure(figsize=(8, 8))
plt.pie(win_counts, labels=win_counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
plt.title('Graph 7: Configuration Dominance (% of Categories where Config was "Best")')
plt.savefig(f"{output_dir}/7_config_win_rate.png")

# 8. Absolute MRR Gain: LLM vs No-LLM (Fixed NER)
mrr_pivot = df_ret.pivot_table(index='category', columns='config', values='mrr')
mrr_pivot['LLM_Lift'] = mrr_pivot['NER:True_LLM:True'] - mrr_pivot['NER:True_LLM:False']
plt.figure(figsize=(12, 6))
mrr_pivot['LLM_Lift'].sort_values().plot(kind='bar', color='green')
plt.title('Graph 8: The "LLM Specialist" MRR Boost (Net Benefit of Specialist Reranking)')
plt.savefig(f"{output_dir}/8_llm_mrr_boost.png", bbox_inches='tight')

# 9. Absolute MRR Gain: NER vs No-NER (Fixed LLM)
mrr_pivot['NER_Lift'] = mrr_pivot['NER:True_LLM:True'] - mrr_pivot['NER:False_LLM:True']
plt.figure(figsize=(12, 6))
mrr_pivot['NER_Lift'].sort_values().plot(kind='bar', color='orange')
plt.title('Graph 9: The "NER Metadata" MRR Boost (Net Benefit of Extraction)')
plt.savefig(f"{output_dir}/9_ner_mrr_boost.png", bbox_inches='tight')

# ---------------------------------------------------------
# SECTION 4: EXTRACTION QUALITY DEEP-DIVE
# ---------------------------------------------------------

# 10. Extraction Efficiency Index (NER count / Fill Rate)
df_ext['efficiency'] = df_ext['ner_tag_count'] / (df_ext['intent_fill_rate'] + 0.1)
plt.figure(figsize=(12, 6))
sns.barplot(x='category', y='efficiency', data=df_ext)
plt.title('Graph 10: Extraction Efficiency Index (NER Tags per Fill-Rate Unit)')
plt.savefig(f"{output_dir}/10_extraction_efficiency.png", bbox_inches='tight')

# 11. Semantic Drift by Category
plt.figure(figsize=(12, 6))
# Low similarity means the model struggled to paraphrase the industry terms correctly.
sns.violinplot(x='category', y='semantic_similarity', data=df_ext, inner="quart")
plt.title('Graph 11: Semantic Drift (Fidelity of paraphrased CVs across industries)')
plt.savefig(f"{output_dir}/11_semantic_drift.png", bbox_inches='tight')

# 12. Information Scarcity vs Retrieval Count
plt.figure(figsize=(10, 6))
merged['fill_bin'] = pd.cut(merged['intent_fill_rate'], bins=3, labels=['Low Fill', 'Med Fill', 'Full Fill'])
sns.barplot(x='fill_bin', y='count_retrieved', hue='config', data=merged)
plt.title('Graph 12: Retrieval Density vs. Data Completeness')
plt.savefig(f"{output_dir}/12_fill_vs_count.png")

# ---------------------------------------------------------
# SECTION 5: ADVANCED METRICS (IR-SPECIFIC)
# ---------------------------------------------------------

# 13. Hit Rate Decay (Slope from Hit@1 to Hit@10)
hit_decay = df_ret.groupby('category')[['hit_at_1', 'hit_at_5', 'hit_at_10']].mean()
plt.figure(figsize=(12, 6))
for cat in hit_decay.index[:5]: # Plotting first 5 for clarity
    plt.plot(['Hit@1', 'Hit@5', 'Hit@10'], hit_decay.loc[cat], marker='o', label=cat)
plt.title('Graph 13: Retrieval Decay Curves (Top-5 Categories)')
plt.legend()
plt.savefig(f"{output_dir}/13_decay_curves.png")

# 14. Precision-Recall Proxy (In-Cat Ratio vs Success)
# Aggregated by category to avoid scatter.
agg_metrics = df_ret.groupby('category')[['in_cat_ratio', 'search_success']].mean()
plt.figure(figsize=(8, 8))
sns.regplot(x='in_cat_ratio', y='search_success', data=agg_metrics)
plt.title('Graph 14: Category-Level Precision vs. Reliability')
plt.savefig(f"{output_dir}/14_precision_vs_reliability.png")

# 15. The "Golden Ratio" - (MRR * In-Category Ratio)
# This rewards categories where the first result is correct AND the rest are relevant.
df_ret['quality_score'] = df_ret['mrr'] * df_ret['in_cat_ratio']
plt.figure(figsize=(12, 6))
df_ret.groupby('category')['quality_score'].mean().sort_values().plot(kind='bar', color='gold')
plt.title('Graph 15: The Golden Ratio (Combined Ranking & Relevance Index)')
plt.savefig(f"{output_dir}/15_golden_ratio.png", bbox_inches='tight')

# ---------------------------------------------------------
# SECTION 6: SYSTEM OVERHEAD / EFFICIENCY
# ---------------------------------------------------------

# 16. Retrieval Density (Count Retrieved vs Category)
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='count_retrieved', data=df_ret)
plt.title('Graph 16: Retrieval Breadth (How many matches does the system find?)')
plt.savefig(f"{output_dir}/16_retrieval_breadth.png", bbox_inches='tight')

# 17. LLM Score Skewness (Is the LLM too generous or too harsh?)
plt.figure(figsize=(10, 6))
for cfg in df_ret['config'].unique():
    sns.kdeplot(df_ret[df_ret['config']==cfg]['llama_top_score'], label=cfg)
plt.title('Graph 17: LLM Score Skew (Distribution of Specialist Judgments)')
plt.legend()
plt.savefig(f"{output_dir}/17_score_skew.png")

# 18. The NER Saturation Point (Aggregated)
# We convert the category intervals to strings to avoid the TypeError
merged['ner_quintile'] = pd.qcut(merged['ner_tag_count'], 5).astype(str)

# Sort them so they appear in numerical order on the chart
ner_order = sorted(merged['ner_quintile'].unique())

plt.figure(figsize=(12, 6))
sns.lineplot(x='ner_quintile', y='mrr', hue='config', data=merged, marker='o', sort=True)
plt.xticks(rotation=15)
plt.title('Graph 18: The NER Saturation Point (MRR gain vs Metadata volume)')
plt.savefig(f"{output_dir}/18_ner_saturation.png", bbox_inches='tight')

# 19. Semantic Similarity vs MRR (Binned)
merged['sim_bin'] = pd.cut(merged['semantic_similarity'], bins=5)
plt.figure(figsize=(12, 6))
sns.barplot(x='sim_bin', y='mrr', data=merged)
plt.title('Graph 19: Impact of Extraction Fidelity (Semantic Sim) on Ranking')
plt.savefig(f"{output_dir}/19_fidelity_impact.png")

# 20. Final Correlation Heatmap (Metric Interaction)
plt.figure(figsize=(12, 10))
corr = df_ret[['search_success', 'hit_at_1', 'mrr', 'in_cat_ratio', 'avg_bge_score', 'llama_top_score', 'consensus_score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Graph 20: Feature Correlation Matrix (How metrics influence each other)')
plt.savefig(f"{output_dir}/20_correlation_matrix.png")

# 21. Search Narrowing Effect (Count Retrieved: Base vs NER)
# Does NER actually help narrow down the search (Precision)?
plt.figure(figsize=(10, 6))
sns.boxenplot(x='config', y='count_retrieved', data=df_ret)
plt.title('Graph 21: Search Narrowing - Does NER/LLM reduce "Noise" in results?')
plt.savefig(f"{output_dir}/21_search_narrowing.png")

# 22. LLM Reranking Swap Rate (Conceptual)
# How often did the LLM change the Top 1 result from the BGE baseline?
# (This requires a bit of logic to see if the top_score belongs to the first result)
df_ret['is_top_match_high_score'] = df_ret['llama_top_score'] > 0.8
plt.figure(figsize=(10, 6))
sns.barplot(x='category', y='is_top_match_high_score', data=df_ret)
plt.title('Graph 22: High-Confidence Specialist Matches by Category')
plt.savefig(f"{output_dir}/22_confidence_by_cat.png")

# 23. Semantic Similarity vs Fill Rate (Efficiency Heatmap)
# Identifying "The Sweet Spot" of extraction
plt.figure(figsize=(10, 8))
# Create a 2D density plot to see where most CVs fall
sns.kdeplot(data=df_ext, x="intent_fill_rate", y="semantic_similarity", cmap="Reds", fill=True)
plt.title('Graph 23: Extraction Quality Density (Consistency of the Extractor)')
plt.savefig(f"{output_dir}/23_extraction_density.png")

print("Second set of 43 inference plots generated successfully.")