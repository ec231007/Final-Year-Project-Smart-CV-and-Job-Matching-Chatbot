import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    df = pd.read_csv(PROJECT_ROOT / 'data' / 'chatbot_metrics.csv')
    print("Files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure CSV files are in the current directory.")
    exit()

# 2. GENERATE TABLES FOR REPORT
print("Generating CSV Reports...")

# Global Statistics Table
global_stats = {
    "Metric": ["Total Test Cases", "Overall Tool Accuracy", "Avg Faithfulness", "Avg Relevance"],
    "Value": [
        len(df),
        f"{(df['tool_accuracy'].mean() * 100):.1f}%",
        f"{df['faithfulness_score'].mean():.2f} / 5",
        f"{df['relevance_score'].mean():.2f} / 5"
    ]
}
pd.DataFrame(global_stats).to_csv('report_global_summary.csv', index=False)

# Scenario Performance Table
scenario_perf = df.groupby('scenario_type').agg({
    'tool_accuracy': 'mean',
    'faithfulness_score': 'mean',
    'relevance_score': 'mean'
}).sort_values(by='tool_accuracy', ascending=False)
scenario_perf.to_csv('report_scenario_metrics.csv')

# 3. VISUALIZATIONS
print("Generating Plots...")
sns.set_theme(style="whitegrid", palette="muted")

# Plot 1: Tool Accuracy by Scenario (Horizontal Bar)
plt.figure(figsize=(10, 6))
plt.title('Tool Selection Accuracy by Scenario', fontsize=14)
plot_data = scenario_perf.sort_values('tool_accuracy', ascending=True)
sns.barplot(data=plot_data, x='tool_accuracy', y=plot_data.index, color='skyblue')
plt.xlabel('Accuracy Rate (1.0 = 100%)', fontsize=12)
plt.ylabel('Scenario Type', fontsize=12)
plt.tight_layout()
plt.savefig('eval_tool_accuracy.png', dpi=300)
plt.clf()

# Plot 2: Quality Scores (Faithfulness vs Relevance)
plt.figure(figsize=(12, 6))
melted_df = scenario_perf.reset_index().melt(
    id_vars='scenario_type', 
    value_vars=['faithfulness_score', 'relevance_score'],
    var_name='Metric', 
    value_name='Score'
)
# Clean up labels for the legend
melted_df['Metric'] = melted_df['Metric'].str.replace('_score', '').str.title()

sns.barplot(data=melted_df, x='Score', y='scenario_type', hue='Metric')
plt.title('Content Quality Scores by Scenario', fontsize=14)
plt.xlim(0, 5) # Scores are on 1-5 scale
plt.xlabel('Average Score (out of 5)', fontsize=12)
plt.ylabel('Scenario Type', fontsize=12)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('eval_quality_scores.png', dpi=300, bbox_inches='tight')
plt.clf()

# Plot 3: Heatmap of Faithfulness (Category vs Scenario)
plt.figure(figsize=(12, 8))
pivot_df = df.pivot_table(
    index='resume_category', 
    columns='scenario_type', 
    values='faithfulness_score', 
    aggfunc='mean'
)

# Handle cases where there might be missing data for a category/scenario combo
if pivot_df.isnull().values.any():
    print("Note: Some Category/Scenario combinations have missing data. They will appear blank in the heatmap.")

sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f', 
            cbar_kws={'label': 'Faithfulness Score (1-5)'},
            linewidths=.5)
plt.title('Heatmap: Faithfulness Scores across Domains', fontsize=14)
plt.xlabel('Scenario Type', fontsize=12)
plt.ylabel('Resume Category', fontsize=12)
# Rotate x labels for better readability if they are long
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('eval_domain_heatmap.png', dpi=300)
plt.clf()

print("✅ Complete! Report artifacts generated in the current directory:")
print("  - report_global_summary.csv")
print("  - report_scenario_metrics.csv")
print("  - eval_tool_accuracy.png")
print("  - eval_quality_scores.png")
print("  - eval_domain_heatmap.png")