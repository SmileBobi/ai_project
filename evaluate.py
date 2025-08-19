import json
import os
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from mylangchain_llm_optim import create_qa_system
from bert_score import score as bert_score

smooth = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ==== 自动评分函数 ====
def evaluate_answer(predicted, ground_truth):
    predicted = predicted.strip()
    ground_truth = ground_truth.strip()

    # Exact Match + Contains
    exact_match = int(predicted.lower() == ground_truth.lower())
    contains_score = int(ground_truth.lower() in predicted.lower() or predicted.lower() in ground_truth.lower())

    # BLEU
    reference = [ground_truth.split()]
    hypothesis = predicted.split()
    bleu = sentence_bleu(reference, hypothesis, smoothing_function=smooth)

    # ROUGE
    rouge = scorer.score(ground_truth, predicted)
    rouge1 = rouge['rouge1'].fmeasure
    rouge2 = rouge['rouge2'].fmeasure
    rougeL = rouge['rougeL'].fmeasure

    # BERTScore (取 F1)
    P, R, F1 = bert_score([predicted], [ground_truth], lang="zh", verbose=False)
    bert_f1 = float(F1[0])

    return {
        "exact_match": exact_match,
        "contains": contains_score,
        "bleu": bleu,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bertscore": bert_f1
    }

# ==== 多模型评估逻辑 ====
def evaluate_model(model_info, questions):
    print(f"\n🧪 正在评估模型: {model_info['name']}")
    qa_system = create_qa_system(
        api_base=model_info["api_base"],
        api_key=model_info["api_key"],
        model_name=model_info["model_name"],
        temperature=0.1
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
            "ground_truth": answer
        })
        model_results.append(metrics)

    return model_results

# ==== 汇总输出 ====
def summarize_report(all_results, save_csv=False):
    from collections import defaultdict
    import csv

    stats = defaultdict(list)
    for r in all_results:
        model = r["model"]
        stats[model].append(r)

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
            "bertscore": avg("bertscore")
        }
        summary.append(row)

        print(f"\n🧠 {model} - 样本数: {total}")
        for k in row.keys():
            if k not in ["model", "samples"]:
                print(f"{k}: {row[k]:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_csv:
        with open("results/report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    return summary


# ==== 主函数 ====
if __name__ == "__main__":
    # 多模型配置
    models = [
        {
            "name": "qwen-plus",
            "model_name": "qwen-plus",
            "api_base": "https://api.fe8.cn/v1",
            "api_key": "sk-8ltHAaRsRD2SMFURe6tsy1lcBySE2BGr67Qg3Ucl7QpiCM9J"
        }
    ]

    with open("data/questions.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    all_model_results = []
    for model in models:
        results = evaluate_model(model, qa_data)
        all_model_results.extend(results)

    summarize_report(all_model_results, save_csv=True)
