from rich import print_json, print
from langchain.prompts import PromptTemplate
from Utils.llm import get_llm
from Utils.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from Utils.gen_JsonOutputParser import gen_JsonOutputParser
from Utils.graph_state import GraphState, KnowledgeTree, KnowledgeNode
from modification_models import ModificationList, ModificationOperation
import yaml
import asyncio
from knowledge_tree_modifier import KnowledgeTreeModifier


logger = setup_logger(__name__)


async def init_eva_k_chain(state: GraphState) -> ModificationList:
    source_doc = state.source_doc
    key_KnowledgeTree = state.knowledge_trees
    setup_file = "setup.yaml"
    try:
        with open(setup_file, encoding='utf-8') as f:
            setup_data = yaml.safe_load(f)
            eva_k_prompt_file = setup_data["graph_config"]["eva_k_prompt"]
            try:
                with open(eva_k_prompt_file, encoding='utf-8') as prompt_file:
                    eva_k_prompt = prompt_file.read()
            except FileNotFoundError:
                logger.error(f"提示词文件未找到: {eva_k_prompt_file}")
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

    eva_k_prompt_template = gen_JsonOutputParser(eva_k_prompt, ModificationList)
    input_2_llm = PromptTemplate(
        input_variables=["source_doc", "key_knowledge_trees"],
        template=eva_k_prompt_template
    )
    parser = JsonOutputParser(pydantic_object=ModificationList)
    eva_k_chain = input_2_llm | get_llm("eva_k_llm") | parser
    result = await eva_k_chain.ainvoke({"source_doc": source_doc, "key_knowledge_trees": key_KnowledgeTree})
    return result


def convert_dict_to_pydantic(data: dict) -> KnowledgeTree:
    """将字典结构的知识树转换为pydantic模型"""
    def process_node(node_dict: dict) -> KnowledgeNode:
        children = node_dict.get('children', {})
        processed_children = [
            process_node(child) 
            for child in children.values()
        ] if children else None
        
        return KnowledgeNode(
            title=node_dict['title'],
            content=node_dict['content'],
            children=processed_children
        )
    
    root_node = process_node(data['root'])
    return KnowledgeTree(root=root_node)


async def run_eva_k_chain(state: GraphState) -> ModificationList:
    """运行eva_k_chain并返回修改列表"""
    result = await init_eva_k_chain(state)
    return result


async def run_eva_k_iterations(state: GraphState) -> GraphState:
    """多次运行eva_k_chain并应用修改"""
    setup_file = "setup.yaml"
    try:
        with open(setup_file, encoding='utf-8') as f:
            setup_data = yaml.safe_load(f)
            init_iterations = setup_data["graph_config"].get("eva_k_times", 2)  # 默认2次迭代
    except yaml.YAMLError as ye:
        logger.error(f"YAML解析失败: {str(ye)} - 文件: {setup_file}", exc_info=True)
        raise

    iterations = init_iterations
    modifier = KnowledgeTreeModifier()                        
    for i in range(iterations):
        modification_list = await run_eva_k_chain(state)
        print(f"\n\n=== 第 {i+1} 次迭代 ===\n\n")
        print(modification_list)
        modified_tree = modifier.modify_knowledge_tree(modification_list, state.knowledge_trees)
        modification_stats = modifier.get_modification_stats()
        if modification_stats['total_modifications'] > 0:
            state.knowledge_trees = modified_tree
            iterations = init_iterations
            print(f"\n修改统计:")
            print(f"总修改数: {modification_stats['total_modifications']}")
            print(f"成功: {modification_stats['success']}")
            if modification_stats['errors']:
                print(f"错误: {modification_stats['errors']}")
        else:
            iterations -= 1
            print(f"\n\n没有修改操作，跳过第 {i+1} 次迭代\n\n")
    return state


async def main():
    print("=== 测试 eva_k_worker ===")
    # 测试1: 初始化GraphState
    source_doc_file = "sample_doc/11、《中华人民共和国电信条例》.txt"
    key_knowledge_trees_file = "sample_doc/test_get_k_output.yaml"
    with open(key_knowledge_trees_file, encoding='utf-8') as f:
        knowledge_trees_content = f.read()
        knowledge_dict = yaml.safe_load(knowledge_trees_content)
        knowledge_trees = convert_dict_to_pydantic(knowledge_dict)
    with open(source_doc_file, encoding='utf-8') as f:
        source_doc_content = f.read()
    test_state = GraphState(source_doc=source_doc_content, source_file=source_doc_file, knowledge_trees=knowledge_trees)
    print(f"\n测试1 - 初始化GraphState成功: {test_state.source_file}, 知识点树: {test_state.knowledge_trees}")
    # 测试2: 调用init_eva_k_chain
    try:
        state = await run_eva_k_iterations(test_state)
        print(f"\n\n测试2 - 最后的graph_state的处理结果\n: {state}")
    except Exception as e:
        print(f"\n\n测试2 - 调用init_eva_k_chain失败: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())