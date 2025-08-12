"""
从GraphState中读取source_doc和knowledge_trees，并将其输入llm进行处理，返回评估结果。
评估内容：是否有遗留的知识点，如果有，返回需要添加的知识点的json格式
目的是确保我们知识点树的完整性。
根据评估结果，使用append_knowledge_node函数将新的知识点添加到knowledge_trees中。
通过setup.yaml文件来获取评估次数，这个评估次数的意思是连续两次通过评估都没有新的知识点添加到knowledge_trees中，才算完成评估。默认值为1次

"""
import yaml
import asyncio
from Utils.graph_state import GraphState, KnowledgeTree, KnowledgeNode
from pydantic import BaseModel
from pydantic.fields import Field
from typing import Optional, List
from langchain_core.output_parsers import JsonOutputParser
from Utils.llm import get_llm
from Utils.logger import setup_logger
import json
from langchain.prompts  import PromptTemplate
from rich import print

logger = setup_logger(__name__)

class MissingNode(BaseModel):
    title: str = Field(
        ...,
        description="知识点标题",
    )
    content: str = Field(
        ...,
        description="知识点详细内容",
    )
    path: List[str] = Field(
        ...,
        description="知识点在知识树中的路径列表，如['总则', '第五条']",
    )

class EvaluationResult(BaseModel):
    status: str = Field(
        ...,
        description="评估状态: 'complete'或'incomplete'",
    )
    point: Optional[MissingNode] = Field(
        default=None,
        description="发现的遗漏知识点，status为'incomplete'时必须提供",
    )

custom_example = """
    完整的JSON输出格式说明：
    
    情况1 - 知识树完整，无遗漏:
    {{
        "status": "complete",
        "points": []
    }}
    
    情况2 - 知识树不完整，有单个遗漏:
    {{
        "status": "incomplete",
        "point": {{
            "title": "第五条", 
            "content": "第五条 电信业务经营者应当为电信用户提供迅速、准确、安全、方便和价格合理的电信服务。",
            "path": ["总则", "第五条"]
        }}
    }}
    
    情况3 - 知识树不完整，有多个不同类型的遗漏:
    {{
        "status": "incomplete",
        "point": {{
            "title": "第五条",
            "content": "第五条 电信业务经营者应当为电信用户提供迅速、准确、安全、方便和价格合理的电信服务。",
            "path": ["中华人民共和国电信条例", "总则", "第五条"]
        }}
    }}
    
    重要提醒：
    1. status字段只能是"complete"或"incomplete"
        - 如果没有遗漏知识点，请务必返回status为"complete"且point为null的json格式。
        - 如果有遗漏知识点，请务必返回status为"incomplete"且point包含遗漏知识点的json格式。
    2. point对象必须包含title、content、path三个字段
    3. path路径必须准确反映知识树的层级结构
    4. 请以严格按照要求的json格式输出，json字段必须按照要求使用英文来表示。不要遗漏和缺失任何要求的json字段。
    5. json格式中不要遗漏title字段。不要把path字段写成target_path，不要把content字段写成missing_content。
    6. 确保json格式正确，避免语法错误，如缺少逗号、引号等。
    7. 避免多余的文本输出，确保返回纯净的json格式。

"""
    
def append_knowledge_node(
    tree: KnowledgeTree,
    target_path: List[str],
    new_node: KnowledgeNode
) -> KnowledgeTree:
    """
    在知识树的指定路径添加子节点
    
    Args:
        tree: 待修改的知识树
        target_path: 目标父节点路径（e.g. ["中华人民共和国电信条例", "总则", "第六条"]）
        new_node: 待添加的知识节点
        
    Returns:
        修改后的知识树（原树未被修改）
    """
    # 复制树结构避免修改原始数据
    new_tree = tree.model_copy(deep=True)
    
    if new_tree.children is None:
        new_tree.children = []
    
    current_node = new_tree
    parent_nodes = []
    
    # 逐层定位目标节点
    for level, path_segment in enumerate(target_path):
        found = False
        
        # 确保当前节点有children列表
        if current_node.children is None:
            current_node.children = []
        
        # 特殊处理顶层节点匹配
        if level == 0:
            # 顶层节点需要与知识树根节点匹配
            if current_node.title == path_segment:
                found = True
                continue
        
        # 在当前节点的子节点中查找
        for child in current_node.children:
            if child.title == path_segment:
                # 找到匹配节点
                parent_nodes.append(child)
                current_node = child
                found = True
                break
        
        # 如果没找到，创建新节点
        if not found:
            # 确保当前节点有children列表
            if current_node.children is None:
                current_node.children = []
            
            # 如果是路径末端节点，直接使用new_node
            if level == len(target_path) - 1:
                # 直接使用new_node，不创建包装节点
                current_node.children.append(new_node)
                return new_tree
                
            # 否则创建中间节点
            new_parent = KnowledgeNode(
                title=path_segment,
                content=f"自动创建的节点: {path_segment}",
                children=[]
            )
            current_node.children.append(new_parent)
            parent_nodes.append(new_parent)
            current_node = new_parent
            
            # 确保新节点的children列表存在
            if current_node.children is None:
                current_node.children = []
    
    # 确保目标节点有children列表
    if current_node.children is None:
        current_node.children = []
    
    # 再次确认children列表存在
    if current_node.children is not None:
        current_node.children.append(new_node)
    else:
        raise ValueError("无法添加节点：目标节点的children列表为None")
    
    return new_tree

async def init_evaluation_chain(state: GraphState) -> KnowledgeTree:
    source_doc = state.source_doc
    knowledge_trees = state.knowledge_trees
    setup_file = "setup.yaml"
    try:
        with open(setup_file, 'r', encoding='utf-8') as f:
            setup_config = yaml.safe_load(f)
            eva_k_times = setup_config.get('graph_config', {}).get('eva_k_times', 1)
            print(f"评估次数阈值: {eva_k_times}")
            get_e_prompt_file = setup_config.get('graph_config', {}).get('eva_k_prompt', 'prompt/eva_k_prompt.md')
            try:
                with open(get_e_prompt_file, 'r', encoding='utf-8') as prompt_file:
                    get_e_prompt = prompt_file.read()
            except FileNotFoundError:
                logger.error(f"找不到配置文件: {get_e_prompt_file}", exc_info=True)
                raise
            except yaml.YAMLError as e:
                logger.error(f"加载{get_e_prompt_file}配置失败: {str(e)}", exc_info=True)
                raise
    except yaml.YAMLError as e:
        logger.error(f"加载{setup_file}配置失败: {str(e)}", exc_info=True)
        raise
    except KeyError as e:
        logger.error(f"配置键缺失: {str(e)} - 文件: {setup_file}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"加载{setup_file}配置失败: {str(e)}", exc_info=True)
        raise   
    
    parser = JsonOutputParser(pydantic_object=EvaluationResult)
    format_instructions = parser.get_format_instructions()
    input_2_llm = PromptTemplate(
        input_variables=["source_doc", "knowledge_trees"],
        template=get_e_prompt+"\n\n"+custom_example,
        partial_variables={"format_instructions": format_instructions}
    )

    e_chain = input_2_llm | get_llm("eva_k_llm") | parser
    
    complete_count = 0
    new_tree = state.knowledge_trees.model_copy(deep=True)
    
    while complete_count < eva_k_times:
        print(f"\n=== 第 {complete_count + 1} 次评估 ===")
        # print("当前知识树版本:")
        # print(json.dumps(json.loads(new_tree.model_dump_json()), indent=2, ensure_ascii=False))
        
        final_result = e_chain.invoke({
            "source_doc": source_doc,
            "knowledge_trees": new_tree  # 使用更新后的知识树进行下一次评估
        })
        
        print(f"\n评估结果 #{complete_count + 1}:")
        print(json.dumps(final_result, indent=2, ensure_ascii=False))
        
        if final_result.get('status') == 'incomplete' and final_result.get('point'):
            # 统一处理单个或多个遗漏知识点
            nodes = final_result['point'] if isinstance(final_result['point'], list) else [final_result['point']]
            
            for node in nodes: 
                new_node = KnowledgeNode(
                    title=node['title'],
                    content=node['content']
                )
                target_path = node['path']
                
                try:
                    new_tree = append_knowledge_node(
                        new_tree,
                        target_path,
                        new_node
                    )
                    print(f"成功添加知识点: {node['title']} -> {target_path}")
                except ValueError as e:
                    print(f"添加知识点失败: {e}")
                    logger.error(f"添加知识点失败: {e}")
            complete_count = 0  # 发现遗漏，重置complete计数
        else:
            complete_count += 1
            print(f"评估通过 ({complete_count}/{eva_k_times})")
    
    print(f"\n评估完成: 连续{eva_k_times}次评估通过")
    return new_tree

if __name__ == "__main__":
    source_doc_file = "sample_doc/11、《中华人民共和国电信条例》.txt"
    test_knowledge_trees_file = "sample_doc\\test_get_k_output.json"
    try:
        with open(source_doc_file, 'r', encoding='utf-8') as f:
            source_doc = f.read()
        with open(test_knowledge_trees_file, 'r', encoding='utf-8') as f:
            knowledge_trees = KnowledgeTree.model_validate(yaml.safe_load(f))
        state = GraphState(
            source_doc=source_doc,
            source_file=source_doc_file,
            knowledge_trees=knowledge_trees
        )
        # print("初始GraphState:")
        # print(state.model_dump_json())
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"加载文件失败: {str(e)}", exc_info=True)
        raise   
    async def main():
        result = await init_evaluation_chain(state)
        return GraphState(
            knowledge_trees=result
        )
    
    result = asyncio.run(main())
    print(f"更新后的知识树: {result}")
    with open("sample_doc/test_eva_k_output.json", "w", encoding='utf-8') as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
