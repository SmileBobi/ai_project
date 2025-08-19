import os
import json
import time
import requests
import torch
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from FlagEmbedding import FlagReranker  # 用于重排序
import re

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 辅助函数 ---
def batch_list(data: list, batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def split_text(text, chunk_size=300, overlap=50):
    """优化切分：优先按句/字段切割，再做滑动窗口"""
    # 按换行和句号分割
    sentences = re.split(r'(?<=[。；！：\n])', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())

    # 滑动窗口补充
    final_chunks = []
    for c in chunks:
        if len(c) <= chunk_size:
            final_chunks.append(c)
        else:
            start = 0
            while start < len(c):
                end = min(len(c), start + chunk_size)
                final_chunks.append(c[start:end])
                start += chunk_size - overlap
    return final_chunks

def merge_results(vector_docs, keyword_docs, top_k=10):
    """合并向量检索和关键词检索结果并去重"""
    all_docs = {doc.page_content: doc for doc in (vector_docs + keyword_docs)}
    return list(all_docs.values())[:top_k]

# --- 核心类 ---
class IntelligentQASystem:
    def __init__(self, bidding_store, enterprise_store, llm, prompt, top_k=10, use_reranker=True):
        self.bidding_retriever = bidding_store.as_retriever(search_kwargs={"k": top_k*2})
        self.enterprise_retriever = enterprise_store.as_retriever(search_kwargs={"k": top_k*2})
        self.llm = llm
        self.prompt = prompt
        self.top_k = top_k
        self.use_reranker = use_reranker
        if use_reranker:
            print("加载重排序模型 bge-reranker-large ...")
            self.reranker = FlagReranker(r"E:\ai\models\bge-reranker-large", use_fp16=True)
        else:
            self.reranker = None

    def expand_query(self, question):
        """生成同义问法 + 关键词抽取"""
        queries = [question]
        try:
            # 同义问法
            prompt = f"请帮我生成3个与以下问题意思相同的不同问法：\n{question}"
            variants = self.llm.invoke(prompt).content.strip().split("\n")
            variants = [v.strip(" -") for v in variants if v.strip()]
            queries.extend(variants)

            # 关键词抽取
            kw_prompt = f"请从以下问题中提取出最核心的关键词（用空格分隔）：\n{question}"
            keywords = self.llm.invoke(kw_prompt).content.strip()
            if keywords and len(keywords) < 30:
                queries.append(keywords)
        except:
            pass
        return list(set(queries))

    def rerank_docs(self, docs, query):
        """对检索结果重排序"""
        if not self.reranker or not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.compute_score(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:self.top_k]]

    def query(self, question):
        try:
            queries = self.expand_query(question)
            bidding_docs, enterprise_docs = [], []

            for q in queries:
                # 向量检索
                vec_bidding = self.bidding_retriever.invoke(q)
                vec_enterprise = self.enterprise_retriever.invoke(q)
                # BM25 检索（加大范围）
                bm25_bidding = BM25Retriever.from_documents(
                    list(self.bidding_retriever.vectorstore.docstore._dict.values()),
                    k=self.top_k*2
                ).invoke(q)
                bm25_enterprise = BM25Retriever.from_documents(
                    list(self.enterprise_retriever.vectorstore.docstore._dict.values()),
                    k=self.top_k*2
                ).invoke(q)
                # 合并结果
                bidding_docs.extend(merge_results(vec_bidding, bm25_bidding, self.top_k*2))
                enterprise_docs.extend(merge_results(vec_enterprise, bm25_enterprise, self.top_k*2))

            # 去重
            bidding_docs = list({doc.page_content: doc for doc in bidding_docs}.values())
            enterprise_docs = list({doc.page_content: doc for doc in enterprise_docs}.values())

            # 重排序
            bidding_docs = self.rerank_docs(bidding_docs, question)
            enterprise_docs = self.rerank_docs(enterprise_docs, question)

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

# --- 构建索引 ---
def rebuild_index(json_path, index_path, embeddings_model):
    print(f"开始重建索引: {index_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    docs = []
    print(f"正在准备文档内容并切分...")
    for item in tqdm(data, desc=f"正在处理 {os.path.basename(json_path)}"):
        content_str = ""
        if isinstance(item, dict):
            for key, value in item.items():
                if value:
                    content_str += f"{key}: {value}\n"
        else:
            content_str = str(item)

        # 切分长文本
        for chunk in split_text(content_str.strip()):
            docs.append(Document(
                page_content=chunk,
                metadata={"source_data": json.dumps(item, ensure_ascii=False)}
            ))

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    batch_size = 64

    print(f"开始批量生成向量 (每批 {batch_size} 条)...")
    all_embeddings = []
    for batch_texts in tqdm(list(batch_list(texts, batch_size)), desc=f"正在嵌入 {os.path.basename(json_path)}"):
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        all_embeddings.extend(batch_embeddings)

    if not all_embeddings:
        raise ValueError("向量嵌入失败。")

    text_embedding_pairs = list(zip(texts, all_embeddings))
    print("向量生成完毕，正在构建FAISS索引...")
    store = FAISS.from_embeddings(text_embedding_pairs, embeddings_model, metadatas=metadatas)
    store.save_local(index_path)
    print(f"索引 {index_path} 重建并保存成功！")
    return store

# --- 创建问答系统 ---
def create_qa_system(api_base=None, api_key=None, model_name="gpt-3.5-turbo", temperature=0.1, top_k=10, **kwargs):
    print("正在初始化本地嵌入模型 BAAI/bge-base-zh-v1.5...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        except:
            print(f"{os.path.basename(index_path)} 索引缺失或损坏，开始重建...")
            rebuild_index(json_path, index_path, embeddings)

    bidding_store = FAISS.load_local(bidding_index_path, embeddings, allow_dangerous_deserialization=True)
    enterprise_store = FAISS.load_local(enterprise_index_path, embeddings, allow_dangerous_deserialization=True)

    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_base=api_base,
        openai_api_key=api_key
    )

    prompt_template = """你是一个专业的招投标采购AI助手。请根据下面提供的“检索到的招标文档”和“检索到的企业文档”中的信息，来全面、准确地回答用户的问题。

请遵循以下规则：
1. 答案必须完全基于提供的上下文信息，禁止编造或使用外部知识。
2. 如果提供的信息足以回答问题，请清晰、有条理地组织答案。
3. 如果提供的信息不包含与问题相关的任何内容，请明确回答：“根据现有资料，无法回答该问题。”
4. 在回答时，不要提及“根据检索到的信息”或“根据上下文”等词语，直接给出答案。

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

    return IntelligentQASystem(bidding_store, enterprise_store, llm, PROMPT, top_k=top_k, use_reranker=True)

# --- 主程序 ---
if __name__ == "__main__":
    API_KEY = "你的API_KEY"
    API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = "qwen-plus-0723"

    print("🚨 欢迎使用智能招投标采购问答系统 🚨")

    qa_system = create_qa_system(
        api_base=API_BASE,
        api_key=API_KEY,
        model_name=MODEL_NAME,
        temperature=0.1,
        top_k=10
    )

    while True:
        user_question = input("请输入您的问题 (输入 '退出' 或 'exit' 来结束): \n> ")
        if user_question.lower() in ["退出", "exit"]:
            print("再见！")
            break
        if not user_question.strip():
            continue
        print("[AI助手] 正在思考中...")
        start_time = time.time()
        answer = qa_system.query(user_question)
        end_time = time.time()
        print(f"\n[AI助手]:\n{answer}")
        print(f"(耗时: {end_time - start_time:.2f} 秒)")
