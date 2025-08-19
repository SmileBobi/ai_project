import requests

# 修正后的配置
API_KEY = "sk-aa9ac612575f4cffb71e8e2e537a4e50"  # ！！！请替换为您的真实有效密钥！！！
API_BASE = "https://api.deepseek.com"  # 填写你的API地址径
MODEL_NAME = "qwen-max"  # 保持模型名称不变

print("正在检查LLM API连接...")
try:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(f"{API_BASE}/models", headers=headers, timeout=10)
    if response.status_code != 200:
        raise ValueError(f"API状态码: {response.status_code} - {response.text}")
    print("LLM API连接成功！")
except Exception as e:
    raise ConnectionError(f"LLM API连接失败: {str(e)}. 请检查API_BASE、API_KEY和网络。")
