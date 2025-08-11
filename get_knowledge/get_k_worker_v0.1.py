from rich import print_json,print
from langchain.prompts import PromptTemplate
from Utils.llm import get_llm
from Utils.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from Utils.gen_JsonOutputParser import gen_JsonOutputParser
from Utils.graph_state import GraphState, KnowledgeTree
import yaml
import asyncio

logger = setup_logger(__name__)

async def init_get_k_chain(state:GraphState) -> GraphState:
    source_doc = state.source_doc
    setup_file= "setup.yaml"
    try:
        with open(setup_file, encoding='utf-8') as f:
            setup_data = yaml.safe_load(f)
            get_k_prompt_file = setup_data["graph_config"]["get_k_prompt"]
            try :
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

    get_k_prompt_template = gen_JsonOutputParser(get_k_prompt, KnowledgeTree)
    input_2_llm = PromptTemplate(
    input_variables = ["source_doc"],
    template = get_k_prompt_template
    )
    parser = JsonOutputParser(pydantic_object=KnowledgeTree)
    get_k_chain = input_2_llm | get_llm("get_k_llm") | parser
    state.knowledge_trees = await get_k_chain.ainvoke({"source_doc": source_doc})
    return state


if __name__ == "__main__":
    print("=== 测试 get_k_worker ===")
    # 测试1: 初始化GraphState
    source_doc_file = "sample_doc/11、《中华人民共和国电信条例》.txt"
    with open(source_doc_file, encoding='utf-8') as f:
        source_doc_content = f.read()
    test_state = GraphState(source_doc=source_doc_content, source_file=source_doc_file)
    print(f"/n测试1 - 初始化GraphState成功: {test_state.source_file}")
    # 测试2: 调用init_get_k_chain
    try:
        async def main():
            result = await init_get_k_chain(test_state)
            return result
            
        result = asyncio.run(main())
        print(f"/n测试2 - 调用init_get_k_chain成功: {result}")
        with open("sample_doc/test_get_k_output.yaml", "w", encoding='utf-8') as f:
            yaml.dump(result.knowledge_trees, f, allow_unicode=True)
    except Exception as e:
        print(f"/n测试2 - 调用init_get_k_chain失败: {str(e)}")
