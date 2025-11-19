import json
import glob

def compute_avg_metrics(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    n = len(data)
    if n == 0:
        return {}
    
    avg_rougeL = sum(item['rougeL'] for item in data) / n
    avg_cosine = sum(item['cosine_similarity'] for item in data) / n
    avg_bleu = sum(item['bleu'] for item in data) / n
    avg_faithfulness = sum(item['faithfulness'] for item in data) / n
    avg_relevance = sum(item['relevance'] for item in data) / n
    
    return {
        'Avg ROUGE-L': round(avg_rougeL, 3),
        'Avg Cosine Sim': round(avg_cosine, 3),
        'Avg BLEU': round(avg_bleu, 3),
        'Avg Faithfulness': round(avg_faithfulness, 3),
        'Avg Relevance': round(avg_relevance, 3)
    }

# Replace with your JSON files
files = {
    'Small': 'results/results_small.json',
    'Medium': 'results/results_medium.json',
    'Large': 'results/results_large.json'
}

summary = {}

for key, file in files.items():
    summary[key] = compute_avg_metrics(file)

# Print table
print("| Setting | Avg ROUGE-L | Avg Cosine Sim | Avg BLEU | Avg Faithfulness | Avg Relevance |")
print("|---------|-------------|----------------|----------|-----------------|---------------|")
for setting, metrics in summary.items():
    print(f"| {setting} | {metrics['Avg ROUGE-L']} | {metrics['Avg Cosine Sim']} | {metrics['Avg BLEU']} | {metrics['Avg Faithfulness']} | {metrics['Avg Relevance']} |")
