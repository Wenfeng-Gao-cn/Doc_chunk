import json
from Utils.graph_state import KnowledgeTree
from get_knowledge.gen_k_chunk_worker import init_chunk_list

def load_knowledge_tree(json_path: str) -> KnowledgeTree:
    """从JSON文件加载知识树"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return KnowledgeTree(**data["knowledge_trees"])

def print_chunk_list(chunk_list):
    """打印切片列表内容"""
    for i, chunk in enumerate(chunk_list.chunks, 1):
        print(f"切片 {i}:")
        print(f"标题: {chunk.title}")
        print(f"内容: {chunk.content}")
        print("元数据:")
        print(f"父节点路径: {chunk.metadata['parent_path']}")
        print(f"完整路径: {chunk.metadata['full_path']}")
        print(f"源文件: {chunk.metadata['source_file']}")
        print("-" * 50)

def main():
    # 加载测试知识树
    json_path = "sample_doc/test_eva_k_output.json"
    knowledge_tree = load_knowledge_tree(json_path)
    
    # 生成切片列表
    chunk_list = init_chunk_list(knowledge_tree, source_file="11、《中华人民共和国电信条例》.txt")
    
    # 打印结果
    print("生成的切片列表:")
    print_chunk_list(chunk_list)

if __name__ == "__main__":
    main()
