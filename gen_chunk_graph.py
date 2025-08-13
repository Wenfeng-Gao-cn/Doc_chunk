import asyncio
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Sequence
import operator
from Utils.logger import setup_logger
from Utils.graph_state import GraphState
from Utils.readfile_2_str import read_file_to_string as read_file_to_str
from get_knowledge.get_k_worker import init_get_k_chain as get_knowledge_tree
from get_knowledge.eva_Omission_k_worker import init_evaluation_chain as evaluate_knowledge_tree
from get_knowledge.gen_k_chunk_worker import gen_knowledge_chunk as generate_knowledge_chunks
from get_knowledge.create_chunk_from_state import VectorDBManager

logger = setup_logger(__name__)

def read_file_node(state: GraphState):
    """节点1: 读取文件内容到graph state"""
    try:
        file_content = read_file_to_str(state.source_file)
        state.source_doc = file_content
        return state
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        raise

async def get_knowledge_node(state: GraphState):
    """节点2: 生成知识树""" 
    try:
        knowledge_tree = await get_knowledge_tree(state)
        state.knowledge_trees = knowledge_tree
        return state
    except Exception as e:
        logger.error(f"生成知识树失败: {e}")
        raise

async def evaluate_knowledge_node(state: GraphState):
    """节点3: 评估知识树完整性"""
    try:
        evaluated_tree = await evaluate_knowledge_tree(state)
        state.knowledge_trees = evaluated_tree
        return state
    except Exception as e:
        logger.error(f"评估知识树失败: {e}")
        raise

async def generate_chunks_node(state: GraphState):
    """节点4: 生成切片列表"""
    try:
        chunks = await generate_knowledge_chunks(state)
        state.chunk_list = chunks
        return state
    except Exception as e:
        logger.error(f"生成切片失败: {e}")
        raise

def create_chunks_node(state: GraphState):
    """节点5: 写入向量数据库"""
    try:
        db_manager = VectorDBManager()
        db_manager.process_state_to_vectordb(state)
        return {"result": True}
    except Exception as e:
        logger.error(f"写入向量数据库失败: {e}")
        return {"result": False}

async def gen_chunk_graph(file_name: str) -> bool:
    """构建并运行事件驱动的工作流"""
    workflow = StateGraph(GraphState)
    
    # 定义节点
    workflow.add_node("read_file", read_file_node)
    workflow.add_node("get_knowledge", get_knowledge_node)
    workflow.add_node("evaluate_knowledge", evaluate_knowledge_node)
    workflow.add_node("generate_chunks", generate_chunks_node)
    workflow.add_node("create_chunks", create_chunks_node)
    
    # 设置边连接
    workflow.add_edge("read_file", "get_knowledge")
    workflow.add_edge("get_knowledge", "evaluate_knowledge")
    workflow.add_edge("evaluate_knowledge", "generate_chunks")
    workflow.add_edge("generate_chunks", "create_chunks")
    
    # 设置入口和出口
    workflow.set_entry_point("read_file")
    workflow.set_finish_point("create_chunks")
    
    # 创建并运行工作流
    app = workflow.compile()
    initial_state = GraphState(source_file=file_name)
    
    try:
        final_result = False
        print("=== 开始执行工作流 ===")
        
        async for event in app.astream_events(
            initial_state,
            version="v2",
            stream_mode=["value"]
        ):
            # 输出事件信息
            # event_type = event["event"]
            node_name = event.get("name", "unknown")
            
            # # print(f"事件类型: {event_type}")
            # # print(f"节点名称: {node_name}")
            
            # if "data" in event:
            #     data = event["data"]
            #     if "input" in data:
            #         print(f"输入数据: {type(data['input'])}")
            #     if "output" in data:
            #         print(f"输出数据: {type(data['output'])}")
            #         # 如果是最后一个节点的输出，检查结果
            #         if node_name == "create_chunks" and isinstance(data["output"], dict):
            #             if "result" in data["output"]:
            #                 final_result = data["output"]["result"]
            #                 print(f"最终结果: {final_result}")
            
            # print("---")
            
        print("=== 工作流执行完成 ===")
        return final_result
        
    except Exception as e:
        logger.error(f"工作流执行失败: {e}")
        print(f"工作流执行失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        file_name = sys.argv[1]
    else:
        file_name = input("请输入要处理的文件名: ")
    
    asyncio.run(gen_chunk_graph(file_name))
