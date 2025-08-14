from typing import Optional
from langchain.prompts import PromptTemplate
from Utils.llm import get_llm_from_list
from Utils.load_setup import load_setup
import asyncio
from Utils.logger import setup_logger
from Utils.retriever_v2 import retrieve,retrieve_with_rerank
from Utils.dicts_2_md import DocumentDisplayFormatter

logger = setup_logger(__name__)

async def run_app(question: str, system_prompt: Optional[str] = None):
    """运行RAG应用并根据stream_mode返回结果"""
    try:
        # 读取配置文件
        config = load_setup()
        chat_prompt_file = config.get('graph_config', {}).get('chat_prompt',"")
        stream_mode = config.get('graph_config', {}).get('stream_mode', True)
        top_k = config.get('reranker_config',{}).get('top_k',3)
           
        try:
            with open(chat_prompt_file, 'r', encoding='utf-8') as f:
                default_system_prompt = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {chat_prompt_file}")
            raise
        except Exception as e:
            logger.error(f"Error reading prompt file: {e}")
            raise        

        if system_prompt is None:
            system_prompt = ""
        prompt_template = system_prompt + default_system_prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 获取LLM
        # llm = get_llm("RAG_chat_llm")
        # retriever = creat_retriver()
        last_error = None
        seq = 0
        llm_model_name=""
        retrive_result = retrieve_with_rerank(question,top_k)
        retrive_result_formater = DocumentDisplayFormatter()
        format_result =retrive_result_formater.to_markdown(retrive_result)
        yield f"<think>\n{str(format_result)}\n</think>\n"    
        while True:
            try:
                llm = get_llm_from_list("RAG_chat_llm", seq)
                llm_model_name = getattr(llm, 'model_name', 'unknown')
                qa_chain = prompt | llm 
                
                if stream_mode:
                    # 流式处理结果，只返回content文字
                    async for token in qa_chain.astream({"question": question, "context":retrive_result}):
                        if hasattr(token, 'content'):
                            yield token.content
                        elif isinstance(token, str):
                            yield token
                else:
                    # 非流式处理，等待完整结果
                    result = await qa_chain.ainvoke({"question": question, "context":retrive_result})
                    if hasattr(result, 'content'):
                        yield result.content
                    elif isinstance(result, str):
                        yield result
                break
            except IndexError:
                if last_error:
                    logger.error(f"所有LLM配置尝试失败，最后一个错误: {str(last_error)}")
                    yield (f"所有LLM配置尝试失败，最后一个错误: {str(last_error)}")
                    raise last_error
                raise
            except Exception as e:
                last_error = e
                logger.warning(f"LLM:【{llm_model_name}】调用失败(seq={seq}): {str(e)}，尝试下一个配置...")
                yield (f"LLM:【{llm_model_name}】调用失败(seq={seq}): {str(e)}\n正在尝试询问其他大模型，请稍后...\n\n")
                seq += 1            
            
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
