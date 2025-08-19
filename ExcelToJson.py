import pandas as pd
import json

# ========== 文件路径 ==========
excel_path = "data/招投标问答200条_增强版.xlsx"  # 替换为你的 Excel 文件路径
output_json_path = "data/questions.json"          # 输出 JSON 文件路径

# ========== 读取 Excel ==========
df = pd.read_excel(excel_path)

# ========== 重命名字段 ==========
df = df.rename(columns={"问题": "question", "答案": "answer"})

# ========== 添加 ID 字段 ==========
df["id"] = [f"Q{str(i+1).zfill(3)}" for i in range(len(df))]

# ========== 提取需要的字段 ==========
questions = df[["question", "answer"]].to_dict(orient="records")

# ========== 写入 JSON 文件 ==========
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"✅ 已成功保存到 {output_json_path}")
