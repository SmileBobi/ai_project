import os
import json
import time
import requests  # 用于API健康检查
import torch  # 用于检测GPU
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import shutil  # 用于自动删除旧索引
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# --- 辅助函数，用于将列表分批 ---
def batch_list(data: list, batch_size: int):
    """将列表切分为指定大小的批次"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# --- 核心类：问答系统 ---
class IntelligentQASystem:
    def __init__(self, bidding_store, enterprise_store, llm, prompt):
        self.bidding_retriever = bidding_store.as_retriever(search_kwargs={"k": 5})
        self.enterprise_retriever = enterprise_store.as_retriever(search_kwargs={"k": 5})
        self.llm = llm
        self.prompt = prompt

    def query(self, question):
        """
        执行一次完整的问答流程
        """
        try:
            # 使用 .invoke() 替代已弃用的 .get_relevant_documents()
            bidding_docs = self.bidding_retriever.invoke(question)
            enterprise_docs = self.enterprise_retriever.invoke(question)

            context_bidding = "\n\n".join([doc.page_content for doc in bidding_docs])
            context_enterprise = "\n\n".join([doc.page_content for doc in enterprise_docs])

            inputs = {
                "context_bidding": context_bidding,
                "context_enterprise": context_enterprise,
                "question": question
            }
            chain = self.prompt | self.llm
            response = chain.invoke(inputs)
            return response.content
        except Exception as e:
            return f"查询失败: {str(e)}. 请检查嵌入模型或API连接。"


# --- 核心函数：重建FAISS索引 ---
def rebuild_index(json_path, index_path, embeddings_model):
    """
    读取JSON文件，创建优化的文档内容，并构建FAISS索引
    """
    print(f"开始重建索引: {index_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    docs = []
    print(f"正在准备更适合嵌入的文档内容...")
    for item in tqdm(data, desc=f"正在处理 {os.path.basename(json_path)}"):
        content_str = ""
        if isinstance(item, dict):
            for key, value in item.items():
                if value is not None and value != '':
                    content_str += f"{key}: {value}\n"
        else:
            content_str = str(item)

        doc = Document(
            page_content=content_str.strip(),
            metadata={"source_data": json.dumps(item, ensure_ascii=False)}
        )
        docs.append(doc)

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    batch_size = 64

    print(f"开始使用本地模型分批生成向量 (每批 {batch_size} 条)...")
    all_embeddings = []
    for batch_texts in tqdm(list(batch_list(texts, batch_size)), desc=f"正在嵌入 {os.path.basename(json_path)}"):
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

    if not all_embeddings:
        raise ValueError("向量嵌入失败，未能生成任何向量。")

    text_embedding_pairs = list(zip(texts, all_embeddings))

    print("向量生成完毕，正在构建FAISS索引...")
    store = FAISS.from_embeddings(text_embedding_pairs, embeddings_model, metadatas=metadatas)
    store.save_local(index_path)
    print(f"索引 {index_path} 重建并保存成功！")
    return store


# --- 核心函数：创建问答系统实例 ---
def create_qa_system(api_base=None, api_key=None, model_name="gpt-3.5-turbo", temperature=0.1, **kwargs):
    """
    整合所有组件，创建并返回一个问答系统实例
    """

    # def check_api_health(base_url, key):
    #     print("正在检查LLM API连接...")
    #     try:
    #         headers = {"Authorization": f"Bearer {key}"}
    #         response = requests.get(f"{base_url}/models", headers=headers, timeout=10)
    #         if response.status_code != 200:
    #             raise ValueError(f"API状态码: {response.status_code} - {response.text}")
    #         print("LLM API连接成功！")
    #     except Exception as e:
    #         raise ConnectionError(f"LLM API连接失败: {str(e)}. 请检查API_BASE、API_KEY和网络。")
    #
    # if api_base:
    #     check_api_health(api_base, api_key)

    print("正在初始化本地嵌入模型 BAAI/bge-base-zh-v1.5...")
    print("（首次运行会下载模型文件，大小约400-500MB，请耐心等待...）")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到嵌入模型将使用的设备: {device}")

    # embeddings = HuggingFaceBgeEmbeddings(
    #     model_name="BAAI/bge-base-zh-v1.5",
    #     model_kwargs={"device": device},
    #     encode_kwargs={"normalize_embeddings": True}
    # )

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="E:\\ai\\models\\bge-base-zh-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("本地嵌入模型加载成功！")

    bidding_index_path = "./data/bidding_index"
    enterprise_index_path = "./data/enterprise_data"
    bidding_json_path = "./data/bidding_index.json"
    enterprise_json_path = "./data/enterprise_data.json"

    for index_path, json_path in [(bidding_index_path, bidding_json_path),
                                  (enterprise_index_path, enterprise_json_path)]:
        try:
            FAISS.load_local(index_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            print(f"{os.path.basename(index_path)} 索引加载成功，无需重建！")
        except Exception as e:
            print(f"{os.path.basename(index_path)} 索引加载失败: {e}，开始重建...")
            rebuild_index(json_path, index_path, embeddings)

    bidding_store = FAISS.load_local(bidding_index_path, embeddings, allow_dangerous_deserialization=True)
    enterprise_store = FAISS.load_local(enterprise_index_path, embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_base=api_base,
        openai_api_key=api_key
    )

    # from dotenv import load_dotenv, find_dotenv
    #
    # _ = load_dotenv(find_dotenv())
    # llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt_template = """你是一个专业的招投标采购AI助手。请根据下面提供的“检索到的招标文档”和“检索到的企业文档”中的信息，来全面、准确地回答用户的问题。

请遵循以下规则：
1.  答案必须完全基于提供的上下文信息，禁止编造或使用外部知识。
2.  如果提供的信息足以回答问题，请清晰、有条理地组织答案。
3.  如果提供的信息不包含与问题相关的任何内容，请明确回答：“根据现有资料，无法回答该问题。”
4.  在回答时，不要提及“根据检索到的信息”或“根据上下文”等词语，直接给出答案。

---
[检索到的招标文档]
{context_bidding}
---
[检索到的企业文档]
{context_enterprise}
---

[用户问题]
{question}

[你的回答]
"""
    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["context_bidding", "context_enterprise", "question"])

    return IntelligentQASystem(bidding_store, enterprise_store, llm, PROMPT)


# --- 主程序入口 ---
if __name__ == "__main__":
    # ========== 配置区域 ==========
    # LLM (大语言模型) 的配置
    # API_KEY = "sk-aa9ac612575f4cffb71e8e2e537a4e50"  # ！！！请替换为您的真实有效密钥！！！
    # API_BASE = "https://api.deepseek.com/v1"  # 填写你的API地址
    # MODEL_NAME = "deepseek-chat"

    # API_KEY = "sk-8c99bbe5be3344a888ddd08d14cc8d65"  # ！！！请替换为您的真实有效密钥！！！
    # API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云 API地址
    # MODEL_NAME = "qwen-max"

    API_KEY = "sk-8c99bbe5be3344a888ddd08d14cc8d65"  # ！！！请替换为您的真实有效密钥！！！
    API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云 API地址 有免费额度
    MODEL_NAME = "qwen-plus-0723"
    # ==============================

    print("\n" + "=" * 80)
    print("🚨 欢迎使用智能招投标采购问答系统")
    print("🚨 重要提示：首次运行或更换数据/模型后，请务必删除旧的FAISS索引目录！")
    print("=" * 80 + "\n")

    # 自动删除旧索引（建议首次运行时取消注释）
    # for path_to_remove in ["./data/bidding_index", "./data/enterprise_data"]:
    #      if os.path.exists(path_to_remove):
    #          shutil.rmtree(path_to_remove)
    #          print(f"已自动删除旧索引目录: {path_to_remove}")

    print(f"使用LLM: {MODEL_NAME}")
    print(f"使用本地嵌入模型: BAAI/bge-base-zh-v1.5\n")

    try:
        qa_system = create_qa_system(
            api_base=API_BASE,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            temperature=0.1,
        )
        print("\n" + "=" * 80)
        print("✅ 系统初始化成功！已进入连续问答模式。")
        print("=" * 80 + "\n")

        # --- 新增功能：循环提问 ---
        while True:
            user_question = input("请输入您的问题 (输入 '退出' 或 'exit' 来结束): \n> ")
            if user_question.lower() in ["退出", "exit"]:
                print("\n感谢使用，再见！")
                break

            if not user_question.strip():
                print("\n[系统提示] 您输入的问题为空，请重新输入。")
                continue

            print("\n[AI助手] 正在思考中，请稍候...")
            start_time = time.time()
            answer = qa_system.query(user_question)
            end_time = time.time()

            print(f"\n[AI助手]:\n{answer}")
            print(f"\n(本次回答耗时: {end_time - start_time:.2f} 秒)")
            print("\n" + "-" * 80 + "\n")

    except Exception as e:
        print(f"\n[致命错误] 系统初始化或运行过程中发生错误: {e}")
        print("程序已终止。请检查配置、网络或依赖项。")