from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from Utils.load_setup import load_setup
from Utils.connect_embeddings import CustomEmbeddings
from Utils.logger import setup_logger
from rich import print
from typing import Optional
from Utils.CustomReranker import CustomReranker


logger = setup_logger(__name__)


def create_retriever():
    """创建带重排功能的检索器"""
    config = load_setup()
    
    # 初始化向量数据库
    embedding_config = config.get('embedding_model', {})
    vector_db = CustomEmbeddings(
        model_name=embedding_config['model_name'],
        api_key=embedding_config['openai_api_key'],
        api_base=embedding_config['openai_api_base']
    )
    
    # 初始化向量存储
    persist_directory_config = config.get('vectordb_config', {}).get('persist_directory')
    collection_name = config.get('vectordb_config', {}).get('collection_name')
    
    vectorstore = Chroma(
        embedding_function=vector_db,
        persist_directory=persist_directory_config,
        collection_name=collection_name
    )

    # 检查collection是否存在且不为空
    collection_info = vectorstore.get()
    if not collection_info.get('ids'):
        logger.warning(f"Collection '{collection_name}' is empty or does not exist!")
    else:
        logger.info(f"Collection '{collection_name}' contains {len(collection_info['ids'])} documents")

    # 创建基础检索器
    retriever_config = config.get('retriever_config', {})
    logger.info(f"Retriever config: {retriever_config}")
    base_retriever = vectorstore.as_retriever(**retriever_config)
    
    # 初始化重排模型
    reranking_config = config.get('reranking_model', {})
    if reranking_config:
        logger.info(f"使用重排模型: {reranking_config.get('model_name')}")
        reranker = CustomReranker(
            model_name=reranking_config['model_name'],
            api_key=reranking_config['openai_api_key'],
            api_base=reranking_config['openai_api_base'],
            max_retries=reranking_config.get('max_retries', 3),
            request_timeout=reranking_config.get('request_timeout', 60)
        )
        
        # 创建带压缩（重排）功能的检索器
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
    else:
        logger.info("未配置重排模型，使用基础检索器")
        retriever = base_retriever
    
    return retriever

def retrieve(question: str):
    """检索并重排文档"""
    retriever = create_retriever()
    results = retriever.invoke(question)
    return results

def retrieve_with_rerank(question: str, top_k: Optional[int] = None):
    """检索并重排文档，支持指定返回数量"""
    retriever = create_retriever()
    results = retriever.invoke(question)
    
    # 如果指定了top_k，则只返回前top_k个结果
    if top_k and len(results) > top_k:
        results = results[:top_k]
    
    return results

if __name__ == "__main__":
    question = "大模型机器人业务流程的FREESWITCH场景有哪些？"
    
    print("🔍 开始检索和重排...")
    result = retrieve(question)
    
    print(f"\n✅ 返回的切片数量：{len(result)}")
    
    # 显示重排分数
    print("\n📊 重排分数:")
    for i, chunk in enumerate(result):
        score = chunk.metadata.get('rerank_score', 'N/A')
        print(f"第{i+1}条: 重排分数 = {score}")
    
    input("\n按任意键打印出所有切片结果内容\n")
    
    for i, chunk in enumerate(result):
        rerank_score = chunk.metadata.get('rerank_score', 'N/A')
        print(f"\n----------第{i+1}条切片 (重排分数: {rerank_score})-------------")
        print(f"内容: {chunk.page_content}")
        print(f"元数据: {chunk.metadata}")
        
    # 测试指定返回数量的功能
    print(f"\n🎯 测试返回前3个最相关的结果:")
    top3_results = retrieve_with_rerank(question, top_k=3)
    for i, chunk in enumerate(top3_results):
        score = chunk.metadata.get('rerank_score', 'N/A')
        print(f"Top{i+1} (分数: {score}): {chunk.page_content[:100]}...")
