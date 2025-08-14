from typing import Optional
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from Utils.llm import get_llm
from get_knowledge.create_chunk_from_state import CustomEmbeddings
import asyncio
import yaml

async def run_app(question: str, system_prompt: Optional[str] = None):
    """运行RAG应用并根据stream_mode返回结果"""
    try:
        # 读取配置文件
        with open('setup.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 获取流式模式配置
        stream_mode = config.get('Match_QA_config', {}).get('stream_mode', True)
        persist_directory_config= config.get('vectordb_config',{}).get('persist_directory')
        chat_prompt_file = config.get('graph_config').get('chat_prompt')
        # 初始化向量数据库
        embedding_config = config['embedding_model']
        vector_db = CustomEmbeddings(
            model_name=embedding_config['model_name'],
            api_key=embedding_config['openai_api_key'],
            api_base=embedding_config['openai_api_base']
        )
        
        # 初始化向量存储
        vectorstore = Chroma(
            embedding_function=vector_db,
            persist_directory=persist_directory_config
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # # 获取系统提示词
        # if system_prompt is None:
        #     with open(chat_prompt_file, 'r', encoding='utf-8') as f:
        #         system_prompt = f.read().strip()
        
        with open(chat_prompt_file, 'r', encoding='utf-8') as f:
            default_system_prompt = f.read().strip()

        if system_prompt is None:
            system_prompt = ""
        prompt_template = system_prompt + default_system_prompt
        # prompt_template = system_prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 获取LLM
        llm = get_llm("RAG_chat_llm")
        llm.streaming = stream_mode
        
        # 构建调用链
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        if stream_mode:
            # 流式处理结果，只返回content文字
            async for token in qa_chain.astream(question):
                if hasattr(token, 'content'):
                    yield token.content
                elif isinstance(token, str):
                    yield token
        else:
            # 非流式处理，等待完整结果
            result = await qa_chain.ainvoke(question)
            if hasattr(result, 'content'):
                yield result.content
            elif isinstance(result, str):
                yield result
            
    except Exception as e:
        yield f"程序运行出错: {e}"

async def main():
    """主函数，调用run_app并打印结果"""
    test_question = "大模型机器人业务流程的FREESWITCH场景有哪些？"
    print("用户问题:", test_question)
    print("系统回答:", end=" ")
    
    async for token in run_app(test_question):
        print(token, end="", flush=True)
    print()  # 换行

if __name__ == "__main__":
    asyncio.run(main())
