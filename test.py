import os
from huggingface_hub import snapshot_download

# 指定保存路径
local_dir = r"E:\ai\models\bge-reranker-large"

# 创建目录（如果不存在）
os.makedirs(local_dir, exist_ok=True)

# 下载模型
snapshot_download(
    repo_id="BAAI/bge-reranker-large",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 直接复制，不用符号链接
    resume_download=True,          # 支持断点续传
    revision="main"
)

print("模型已下载到:", local_dir)
