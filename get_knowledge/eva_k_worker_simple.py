"""
简化版本的eva_k_worker
直接使用JSON处理，避免复杂的Pydantic模型嵌套
"""

from rich import print_json, print
from langchain.prompts import PromptTemplate
from Utils.llm import get_llm
from Utils.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from Utils.graph_state import GraphState, KnowledgeTree, KnowledgeNode
import yaml
import asyncio
import json
from typing import Dict, List, Any


logger = setup_logger(__name__)


async def init_eva_k_chain_simple(state: GraphState) -> Dict[str, Any]:
    """简化版本的eva_k_chain，直接返回JSON字典"""
    source_doc = state.source_doc
    key_KnowledgeTree = state.knowledge_trees
    
    # 读取配置
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
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}")
        raise

    # 简化的提示词模板
    simple_prompt = eva_k_prompt + """

请输出JSON格式的修改指令，格式如下：
{
  "modifications": [
    {
      "action": "add|del|modify|none",
      "path": "目标路径",
      "content": {
        "title": "标题",
        "content": "内容",
        "children": null
      },
      "reason": "修改原因"
    }
  ]
}

注意：
1. action为"del"时，content可以为null
2. action为"none"时表示无需修改
3. 只输出JSON，不要其他文字
"""

    input_2_llm = PromptTemplate(
        input_variables=["source_doc", "key_knowledge_trees"],
        template=simple_prompt
    )
    
    # 使用简单的JSON解析器
    parser = JsonOutputParser()
    eva_k_chain = input_2_llm | get_llm("eva_k_llm") | parser
    
    result = await eva_k_chain.ainvoke({
        "source_doc": source_doc, 
        "key_knowledge_trees": key_KnowledgeTree
    })
    
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


async def run_eva_k_chain_simple(state: GraphState) -> Dict[str, Any]:
    """运行简化版eva_k_chain并返回修改字典"""
    result = await init_eva_k_chain_simple(state)
    return result


def apply_modifications_simple(modifications: List[Dict[str, Any]], knowledge_tree: KnowledgeTree) -> KnowledgeTree:
    """应用简化的修改操作到知识树"""
    # 这里可以实现简化的修改逻辑
    # 为了演示，暂时返回原始知识树
    logger.info(f"应用 {len(modifications)} 个修改操作")
    for mod in modifications:
        logger.info(f"修改操作: {mod['action']} - {mod['reason']}")
    return knowledge_tree


async def run_eva_k_iterations_simple(state: GraphState) -> GraphState:
    """简化版本的多次迭代"""
    setup_file = "setup.yaml"
    try:
        with open(setup_file, encoding='utf-8') as f:
            setup_data = yaml.safe_load(f)
            init_iterations = setup_data["graph_config"].get("eva_k_times", 2)
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}")
        raise

    iterations = init_iterations
    
    for i in range(iterations):
        try:
            result = await run_eva_k_chain_simple(state)
            print(f"\n=== 第 {i+1} 次迭代 ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # 检查是否有修改操作
            modifications = result.get('modifications', [])
            active_modifications = [m for m in modifications if m.get('action') != 'none']
            
            if active_modifications:
                print(f"\n发现 {len(active_modifications)} 个修改操作")
                # 应用修改（这里可以实现具体的修改逻辑）
                state.knowledge_trees = apply_modifications_simple(active_modifications, state.knowledge_trees)
                iterations = init_iterations  # 重置迭代次数
            else:
                iterations -= 1
                print(f"\n没有修改操作，跳过第 {i+1} 次迭代")
                
        except Exception as e:
            logger.error(f"第 {i+1} 次迭代失败: {str(e)}")
            break
            
    return state


async def main():
    print("=== 测试简化版 eva_k_worker ===")
    
    # 测试1: 初始化GraphState
    source_doc_file = "sample_doc/11、《中华人民共和国电信条例》.txt"
    key_knowledge_trees_file = "sample_doc/test_get_k_output.yaml"
    
    try:
        with open(key_knowledge_trees_file, encoding='utf-8') as f:
            knowledge_trees_content = f.read()
            knowledge_dict = yaml.safe_load(knowledge_trees_content)
            knowledge_trees = convert_dict_to_pydantic(knowledge_dict)
            
        with open(source_doc_file, encoding='utf-8') as f:
            source_doc_content = f.read()
            
        test_state = GraphState(
            source_doc=source_doc_content, 
            source_file=source_doc_file, 
            knowledge_trees=knowledge_trees
        )
        
        print(f"\n测试1 - 初始化GraphState成功: {test_state.source_file}")
        
        # 测试2: 调用简化版eva_k_chain
        state = await run_eva_k_iterations_simple(test_state)
        print(f"\n测试2 - 处理完成")
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        logger.error(f"测试失败: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
