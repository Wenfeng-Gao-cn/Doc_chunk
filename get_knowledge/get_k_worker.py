from rich import print_json, print
from langchain.prompts import PromptTemplate
from Utils.llm import get_llm
from Utils.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from Utils.graph_state import GraphState,KnowledgeTree,KnowledgeNode
import yaml
import asyncio
import json

logger = setup_logger(__name__)

async def init_get_k_chain(state: GraphState) -> KnowledgeTree:
    source_doc = state.source_doc
    setup_file = "setup.yaml"
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
    input_2_llm = PromptTemplate(
        input_variables=["source_doc"],
        template=get_k_prompt,
        partial_variables={"format_instructions": format_instructions}
    )
    # print("输入模型的内容:")
    # print(json.dumps(input_2_llm.dict(), indent=2, ensure_ascii=False))
    # input("请确认输入模型的内容是否正确，按Enter继续...")

    k_chain = input_2_llm | get_llm("get_k_llm") | parser
    final_result = k_chain.invoke({"source_doc": source_doc})
    # final_result = KnowledgeTree(**result)

    print(f"生成知识树：\n{final_result}\n")

    # return final_result
    # return final_result

    # 确保返回结果包含必需字段并转换为KnowledgeTree实例
    if isinstance(final_result, dict):
        if 'title' not in final_result:
            final_result['title'] = "知识树"
        if 'content' not in final_result:
            final_result['content'] = "这是一个结构化的知识树，包含了相关主题的详细知识点。"
        return KnowledgeTree(**final_result)
    elif isinstance(final_result, KnowledgeTree):
        return final_result
    else:
        raise ValueError(f"无法处理的结果类型: {type(final_result)}")
    

if __name__ == "__main__":
    print("=== 测试 get_k_worker ===")
    
    # 测试1: 初始化GraphState
    source_doc_file = "sample_doc/云趣运维文档1754623245145/语音机器人安装手册V1.7.pdf.txt"
    try:
        with open(source_doc_file, encoding='utf-8') as f:
            source_doc_content = f.read()
        test_state = GraphState(source_doc=source_doc_content, source_file=source_doc_file)
        print(f"\n测试1 - 初始化GraphState成功: {test_state.source_file}")
    except FileNotFoundError:
        print(f"\n测试1 - 文件未找到: {source_doc_file}")
        # # Create a simple test case with the content from your paste
        # source_doc_content = "中华人民共和国电信条例示例内容"
        # test_state = GraphState(source_doc=source_doc_content, source_file="test_content")
        # print(f"\n测试1 - 使用测试内容初始化GraphState成功")
    
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
                json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
            print("✅ 输出文件已保存: sample_doc/test_get_k_output.json")
            
        except Exception as save_error:
            print(f"❌ 保存文件时出错: {str(save_error)}")
            print(f"尝试直接打印结果内容:")
            print(result)
            
    except Exception as e:
        print(f"\n测试2 - 调用init_get_k_chain失败: {str(e)}")
        import traceback
        traceback.print_exc()
