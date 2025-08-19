# ========================
# 文件：evaluate.py（针对召回率评估的修订）
# 关键改动：
# - 将 ROUGE 使用 recall（而非 f1）评估覆盖率
# - 输出更清晰的召回指标，便于对照 89% 目标
# ========================

import json
import os
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

# 注意：按你的实际文件名导入
from mylangchain_llm_optim_optimize import create_qa_system

smooth = SmoothingFunction().method1
# use_stemmer=True 对英文更有效；中文可保持 False 或使用分词后再算
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)


def evaluate_answer(predicted, ground_truth):
    predicted = (predicted or "").strip()
    ground_truth = (ground_truth or "").strip()

    exact_match = int(predicted.lower() == ground_truth.lower())
    contains_score = int(ground_truth.lower() in predicted.lower() or predicted.lower() in ground_truth.lower())

    reference = [ground_truth.split()]
    hypothesis = predicted.split()
    bleu = sentence_bleu(reference, hypothesis, smoothing_function=smooth) if hypothesis else 0.0

    rouge = scorer.score(ground_truth, predicted)
    # 采用召回率（recall）作为“知识召回率”的主要指标
    rouge1 = rouge['rouge1'].recall
    rouge2 = rouge['rouge2'].recall
    rougeL = rouge['rougeL'].recall

    P, R, F1 = bert_score([predicted], [ground_truth], lang="zh", verbose=False)
    bert_f1 = float(F1[0])

    return {
        "exact_match": exact_match,
        "contains": contains_score,
        "bleu": bleu,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bertscore": bert_f1,
    }


def evaluate_model(model_info, questions):
    print(f"\n🧪 正在评估模型: {model_info['name']}")
    qa_system = create_qa_system(
        api_base=model_info["api_base"],
        api_key=model_info["api_key"],
        model_name=model_info["model_name"],
        temperature=0.1,
        top_k=10
    )

    model_results = []
    for item in tqdm(questions, desc=f"模型 {model_info['name']}"):
        question = item["question"]
        answer = item["answer"]
        print(f"答案：{answer}")
        predicted = qa_system.query(question)
        print(f"预测：{predicted}")
        metrics = evaluate_answer(predicted, answer)
        metrics.update({
            "model": model_info["name"],
            "question": question,
            "predicted": predicted,
            "ground_truth": answer,
        })
        model_results.append(metrics)
    return model_results


def summarize_report(all_results, save_csv=False):
    from collections import defaultdict
    import csv

    stats = defaultdict(list)
    for r in all_results:
        stats[r["model"]].append(r)

    print("\n📊 模型评估汇总：")
    summary = []
    for model, records in stats.items():
        total = len(records)
        avg = lambda k: sum(r[k] for r in records) / total
        row = {
            "model": model,
            "samples": total,
            "exact_match": avg("exact_match"),
            "contains": avg("contains"),
            "bleu": avg("bleu"),
            "rouge1": avg("rouge1"),
            "rouge2": avg("rouge2"),
            "rougeL": avg("rougeL"),
            "bertscore": avg("bertscore"),
        }
        summary.append(row)

        print(f"\n🧠 {model} - 样本数: {total}")
        for k in ["exact_match","contains","rouge1","rouge2","rougeL","bleu","bertscore"]:
            print(f"{k}: {row[k]:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_csv and summary:
        import csv
        with open("results/report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    return summary


if __name__ == "__main__":
    models = [
        {
            "name": "deepseek-chat",
            "model_name": "deepseek-chat",
            "api_base": "https://api.deepseek.com/v1",
            "api_key": "sk-aa9ac612575f4cffb71e8e2e537a4e50"
        }
    ]

    with open("data/questions.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    all_model_results = []
    for m in models:
        all_model_results.extend(evaluate_model(m, qa_data))

    summarize_report(all_model_results, save_csv=True)