from rich import print_json, print
from langchain.prompts import PromptTemplate
from Utils.llm import get_llm
from Utils.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from Utils.gen_JsonOutputParser import gen_JsonOutputParser
from Utils.graph_state import GraphState,KnowledgeTree,KnowledgeNode
import yaml
import asyncio
import json

logger = setup_logger(__name__)

async def init_get_k_chain(state: GraphState) -> KnowledgeTree:
    source_doc = state.source_doc
    setup_file = "setup.yaml"
    # KnowledgeNode.model_rebuild()
    # KnowledgeTree.model_rebuild()
    try:
        with open(setup_file, encoding='utf-8') as f:
            setup_data = yaml.safe_load(f)
            get_k_prompt_file = setup_data["graph_config"]["get_k_prompt"]
            try:
                with open(get_k_prompt_file, encoding='utf-8') as prompt_file:
                    get_k_prompt = prompt_file.read()
            except FileNotFoundError:
                logger.error(f"提示词文件未找到: {get_k_prompt_file}")
                raise

    except yaml.YAMLError as ye:
        logger.error(f"YAML解析失败: {str(ye)} - 文件: {setup_file}", exc_info=True)
        raise
    except KeyError as ke:
        logger.error(f"配置键缺失: {str(ke)} - 文件: {setup_file}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"加载setup.yaml配置失败: {str(e)} - 文件: {setup_file}", exc_info=True)
        raise
    parser = JsonOutputParser(pydantic_object=KnowledgeTree)
    format_instructions = parser.get_format_instructions()
    # get_k_prompt_template = gen_JsonOutputParser(get_k_prompt, KnowledgeTree)
    input_2_llm = PromptTemplate(
        input_variables=["source_doc"],
        # template=get_k_prompt_template,
        template=get_k_prompt,
        # partial_variables={"format_instructions": parser.get_format_instructions()}
        partial_variables={"format_instructions": format_instructions}
    )
    print("输入模型的内容:")
    print(json.dumps(input_2_llm.dict(), indent=2, ensure_ascii=False))
    input("请确认输入模型的内容是否正确，按Enter继续...")

    k_chain = input_2_llm | get_llm("get_k_llm") | parser

    # Collect all chunks from the stream with real-time printing
    output_chunks = []
    print("开始流式处理...", flush=True)
    
    async for chunk in k_chain.astream({"source_doc": source_doc}):
        # 实时打印每个chunk，保持流式效果
        if chunk:  # 只打印非空chunk
            # print(f"接收到数据块: {chunk}", flush=True)
            print(chunk, end="", flush=True)
            output_chunks.append(chunk)
        else:
            print(".", end="", flush=True)  # 空chunk时打印点号表示进度
    
    print(f"\n流式处理完成，共接收到 {len(output_chunks)} 个数据块", flush=True)
    
    # 处理流式输出结果
    if not output_chunks:
        raise ValueError("没有接收到任何数据")
    
    # 获取最终完整结果
    final_result = output_chunks[-1] if output_chunks else {}
    print(f"最终结果类型: {type(final_result)}", flush=True)
    
    return final_result
    # 创建KnowledgeTree实例
    # try:
    #     if isinstance(final_result, dict):
    #         # 检查并补充缺失的必需字段（简化版本）
    #         if 'title' not in final_result:
    #             print("警告: 结果中缺少title字段，正在自动生成...", flush=True)
    #             final_result['title'] = "知识树"
            
    #         if 'content' not in final_result:
    #             print("警告: 结果中缺少content字段，正在自动生成...", flush=True)
    #             final_result['content'] = "这是一个结构化的知识树，包含了相关主题的详细知识点。"
            
    #         knowledge_tree = KnowledgeTree(**final_result)
    #         print("成功创建KnowledgeTree实例", flush=True)
    #         return knowledge_tree
    #     elif isinstance(final_result, KnowledgeTree):
    #         print("结果已经是KnowledgeTree实例", flush=True)
    #         return final_result
    #     else:
    #         # 如果结果不是dict也不是KnowledgeTree，尝试解析
    #         print(f"尝试解析非标准结果: {type(final_result)}", flush=True)
    #         parsed_result = parser.parse(str(final_result))
    #         if isinstance(parsed_result, dict):
    #             # 对解析后的结果也进行字段检查
    #             if 'title' not in parsed_result:
    #                 parsed_result['title'] = "知识树"
                
    #             if 'content' not in parsed_result:
    #                 parsed_result['content'] = "这是一个结构化的知识树，包含了相关主题的详细知识点。"
                
    #             return KnowledgeTree(**parsed_result)
    #         return parsed_result
    # except Exception as parse_error:
    #     logger.error(f"解析结果时出错: {str(parse_error)}")
    #     print(f"解析错误，原始结果: {final_result}")
    #     raise


if __name__ == "__main__":
    print("=== 测试 get_k_worker ===")
    
    # 测试1: 初始化GraphState
    source_doc_file = "sample_doc/11、《中华人民共和国电信条例》.txt"
    try:
        with open(source_doc_file, encoding='utf-8') as f:
            source_doc_content = f.read()
        test_state = GraphState(source_doc=source_doc_content, source_file=source_doc_file)
        print(f"\n测试1 - 初始化GraphState成功: {test_state.source_file}")
    except FileNotFoundError:
        print(f"\n测试1 - 文件未找到: {source_doc_file}")
        # Create a simple test case with the content from your paste
        source_doc_content = "中华人民共和国电信条例示例内容"
        test_state = GraphState(source_doc=source_doc_content, source_file="test_content")
        print(f"\n测试1 - 使用测试内容初始化GraphState成功")
    
    # 测试2: 调用init_get_k_chain
    try:
        async def main():
            result = await init_get_k_chain(test_state)
            return result
            
        result = asyncio.run(main())
        print(f"\n测试2 - 调用init_get_k_chain成功!")
        print(f"返回结果类型: {type(result)}")
        
        # # 如果是KnowledgeTree实例，展示一些基本信息
        # if hasattr(result, 'title'):
        #     print(f"知识树标题: {result.title if result.title else 'N/A'}")
        #     if result.children is not None:
        #         print(f"子节点数量: {len(result.children)}")
        
        # Save the result properly
        try:
            # # Check if result is a KnowledgeTree instance
            # if hasattr(result, 'model_dump'):
            #     print("使用model_dump()方法序列化结果...")
            #     output_data = result.model_dump(mode="json")
            #     print("序列化成功!")
            # else:
            #     # If it's already a dict, use it directly
            #     print("结果已经是字典格式，直接使用...")
            #     output_data = result if isinstance(result, dict) else str(result)
            
            with open("sample_doc/test_get_k_output.json", "w", encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print("✅ 输出文件已保存: sample_doc/test_get_k_output.json")
            
        except Exception as save_error:
            print(f"❌ 保存文件时出错: {str(save_error)}")
            print(f"尝试直接打印结果内容:")
            print(result)
            
    except Exception as e:
        print(f"\n测试2 - 调用init_get_k_chain失败: {str(e)}")
        import traceback
        traceback.print_exc()
