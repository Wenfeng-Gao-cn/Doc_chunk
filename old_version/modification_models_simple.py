"""
简化的修改操作模型定义
直接使用字典结构，避免复杂的Pydantic嵌套
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, Any, List, Dict


class SimpleModificationOperation(BaseModel):
    """简化的单个修改操作模型"""
    action: Literal['add', 'del', 'modify', 'none'] = Field(
        ...,
        description="修改类型：添加(add)/删除(del)/修改(modify)/无需修改(none)"
    )
    
    path: str = Field(
        ...,
        description="目标节点路径，如'root.children.chapter1.children'"
    )
    
    content: Optional[Dict[str, Any]] = Field(
        None,
        description="修改内容，包含title、content、children字段"
    )
    
    reason: str = Field(
        ...,
        description="修改原因的简要说明"
    )


class SimpleModificationList(BaseModel):
    """简化的修改操作列表"""
    modifications: List[SimpleModificationOperation] = Field(
        default_factory=list,
        description="修改操作列表"
    )
