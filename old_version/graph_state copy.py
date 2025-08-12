from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class KnowledgeNode(BaseModel):
    """表示单个知识点节点的模型"""
    title: str = Field(..., description="知识点标题")
    content: str = Field(..., description="知识点详细内容")
    children: Optional[List['KnowledgeNode']] = Field(
        default=None, 
        description="子知识点列表"
    )

class KnowledgeTree(BaseModel):
    """表示完整知识点树的模型"""
    root: KnowledgeNode = Field(..., description="知识点树的根节点")

class GraphState(BaseModel):
    """LangGraph 的完整状态定义"""
    knowledge_trees: KnowledgeTree = Field(
        default_factory=lambda: KnowledgeTree(root=KnowledgeNode(title="", content="")),
        description="完整的知识体系的知识点树"
    )
    source_doc: str = Field(
        default="",
        description="源文档的内容"
    )
    source_file: str = Field(
        default="", 
        description="源文档的文件名"
    )
    chunk_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="文本块处理结果列表"
    )

    
# 解决 KnowledgeNode 的前向引用问题
KnowledgeNode.model_rebuild()
