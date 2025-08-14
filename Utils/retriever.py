from langchain_chroma import Chroma
from Utils.load_setup import load_setup
from Utils.connect_embeddings import CustomEmbeddings
from Utils.logger import setup_logger
from rich import print

logger = setup_logger(__name__)

def creat_retriever():
    config = load_setup()
    # 初始化向量数据库
    embedding_config = config.get('embedding_model',{})
    vector_db = CustomEmbeddings(
        model_name=embedding_config['model_name'],
        api_key=embedding_config['openai_api_key'],
        api_base=embedding_config['openai_api_base']
    )
    # 初始化向量存储
    persist_directory_config = config.get('vectordb_config',{}).get('persist_directory')
    collection_name = config.get('vectordb_config',{}).get('collection_name')
    
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

    retriever_config = config.get('retriever_config', {})
    logger.info(f"Retriever config: {retriever_config}")
    retriever = vectorstore.as_retriever(**retriever_config)
    return retriever

def retrieve(question:str):
    retriever = creat_retriever()
    results = retriever.invoke(question)
    return results
    

if __name__ == "__main__":
    question ="大模型机器人业务流程的FREESWITCH场景有哪些？"
    result = retrieve(question)
    print(f"\n返回的切片数量：{len(result)}")
    input("\n按任意键打印出所有切片结果内容\n")
    i=1
    for chunk in result:
        print(f"\n----------第{i}条切片的内容是-------------\n{chunk}")
        i +=1          
    # print(result)
 