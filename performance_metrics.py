import os
import json
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm

# 1️⃣ Load reference text (ocr_result.txt)
with open('data/ocr_result.txt', 'r', encoding='utf-8') as f:
    reference_text = f.read().strip()

# 2️⃣ Prepare list of prompt json files
llm_prompts_dir = 'llm_prompts'
prompt_files = [f for f in os.listdir(llm_prompts_dir) if f.endswith('.json')]

# 3️⃣ Initialize ROUGE scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# 4️⃣ Prepare CSV output
output_file = 'data/performance_metrics_fn.csv'

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['prompt_name', 'BLEU', 'ROUGE1', 'ROUGEL', 'BERTScore']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # 5️⃣ Process each prompt file
    for prompt_file in tqdm(prompt_files, desc='Processing prompts'):
        prompt_name = os.path.splitext(prompt_file)[0]

        with open(os.path.join(llm_prompts_dir, prompt_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            response_text = data.get('response', '').strip()

        # BLEU score
        reference_tokens = reference_text.split()
        response_tokens = response_text.split()
        smoothie = SmoothingFunction().method4
        bleu_score = sentence_bleu([reference_tokens], response_tokens, smoothing_function=smoothie)

        # ROUGE scores
        rouge_scores = rouge_scorer.score(reference_text, response_text)
        rouge1_score = rouge_scores['rouge1'].fmeasure
        rougel_score = rouge_scores['rougeL'].fmeasure

        # BERTScore
        P, R, F1 = bert_score([response_text], [reference_text], lang='en', verbose=False)
        bertscore_f1 = F1[0].item()

        # Write row to CSV
        writer.writerow({
            'prompt_name': prompt_name,
            'BLEU': bleu_score,
            'ROUGE1': rouge1_score,
            'ROUGEL': rougel_score,
            'BERTScore': bertscore_f1
        })

print(f"\n✅ Performance metrics saved to {output_file}")