from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# 知识树模型定义
class KnowledgeNode(BaseModel):
    """表示单个知识点节点的模型"""
    title: str = Field(..., description="知识点的标题")
    content: str = Field(..., description="知识点的内容概括")
    children: Optional[List['KnowledgeNode']] = Field(
        default=None, 
        description="子知识点列表，如果没有子节点则为null"
    )
    
    class Config:
        # 允许前向引用
        arbitrary_types_allowed = True

class KnowledgeTree(BaseModel):
    """表示完整知识点树的模型"""
    title: str = Field(..., description="知识树的主题标题")
    content: str = Field(..., description="知识树的整体概括内容")
    children: Optional[List[KnowledgeNode]] = Field(
        default=None, 
        description="知识点列表"
    )
    
    def get_all_nodes(self, parent_path: List[str] = []) -> List[Dict[str, Any]]:
        """获取所有节点及其父节点路径"""
        all_nodes = []
        if self.children:
            for node in self.children:
                current_path = parent_path + [node.title]
                all_nodes.append({
                    "node": node,
                    "path": current_path
                })
                if node.children:
                    all_nodes.extend(self._get_all_nodes_from_node(node, current_path))
        return all_nodes
    
    def _get_all_nodes_from_node(self, node: KnowledgeNode, parent_path: List[str]) -> List[Dict[str, Any]]:
        """递归获取节点下的所有节点"""
        all_nodes = []
        if node.children:
            for child in node.children:
                current_path = parent_path + [child.title]
                all_nodes.append({
                    "node": child,
                    "path": current_path
                })
                if child.children:
                    all_nodes.extend(self._get_all_nodes_from_node(child, current_path))
        return all_nodes
    
    class Config:
        # 允许前向引用
        arbitrary_types_allowed = True

class KnowledgeChunk(BaseModel):
    """知识切片的模型"""
    title: str = Field(..., description="知识切片的标题")
    content: str = Field(..., description="知识切片的内容")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="与知识切片相关的元数据"
    )

class ChunkList(BaseModel):
    """知识切片列表的模型"""
    chunks: List[KnowledgeChunk] = Field(
        default_factory=list,
        description="知识切片的列表"
    )

class GraphState(BaseModel):
    """LangGraph 的完整状态定义"""
    knowledge_trees: KnowledgeTree = Field(
        default_factory=lambda: KnowledgeTree(
            title="",
            content=""
        ),
        description="知识体系树"
    )
    source_doc: str = Field(
        default="",
        description="源文档的内容"
    )
    source_file: str = Field(
        default="", 
        description="源文档的文件名"
    )
    chunk_list: ChunkList = Field(
        default_factory=lambda: ChunkList(
            chunks=[
                KnowledgeChunk(
                    title="",
                    content="",
                    metadata=None
                )
            ]
        ),
        description="知识切片列表"
    )

    
# 解决 KnowledgeNode 的前向引用问题
KnowledgeNode.model_rebuild()
