"""
修改操作模型定义 - 简化版本，去掉operations包装
包含详细的Pydantic模型描述来约束LLM的JSON输出
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional, Any, List, Dict, Union


class KnowledgeNodeContent(BaseModel):
    """知识节点内容模型"""
    title: str = Field(
        ...,
        description="节点标题，如'第一条'、'第一章 总则'等",
        min_length=1,
        max_length=100
    )
    
    content: str = Field(
        ...,
        description="节点内容描述",
        min_length=1,
        max_length=2000
    )
    
    children: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="子节点列表，如果没有子节点则为null"
    )


class ModificationOperation(BaseModel):
    """单个修改操作模型 - 详细约束版本"""
    action: Literal['add', 'del', 'modify', 'none'] = Field(
        ...,
        description="修改操作类型。必须是以下值之一：'add'(添加新节点)、'del'(删除节点)、'modify'(修改现有节点)、'none'(无需修改)"
    )
    
    path: str = Field(
        ...,
        description="目标节点的完整路径。格式为'root.children[索引]'或'root.children[索引].children[索引]'等。例如：'root.children[0]'表示第一个子节点，'root.children[0].children[1]'表示第一个子节点的第二个子节点",
        min_length=1,
        max_length=200
    )
    
    content: Optional[KnowledgeNodeContent] = Field(
        None,
        description="修改的具体内容。当action为'add'或'modify'时必须提供，当action为'del'或'none'时应为null"
    )
    
    reason: str = Field(
        ...,
        description="执行此修改操作的原因说明",
        min_length=5,
        max_length=200
    )
    
    @validator('content')
    def validate_content(cls, v, values):
        """验证content字段的合理性"""
        action = values.get('action')
        if action in ['add', 'modify'] and v is None:
            raise ValueError(f"当action为'{action}'时，content不能为null")
        if action in ['del', 'none'] and v is not None:
            raise ValueError(f"当action为'{action}'时，content应为null")
        return v
    
    @validator('path')
    def validate_path(cls, v):
        """验证路径格式"""
        if not v.startswith('root'):
            raise ValueError("路径必须以'root'开头")
        return v

    class Config:
        """Pydantic配置 - 加强JSON格式约束"""
        json_schema_extra = {
            "description": "知识树修改操作。必须严格按照此格式输出JSON数组",
            "type": "object",
            "required": ["action", "path", "reason"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "del", "modify", "none"],
                    "description": "操作类型，必须是add、del、modify或none之一"
                },
                "path": {
                    "type": "string",
                    "pattern": "^root(\.children\[\d+\])*$",
                    "description": "节点路径，必须以root开头"
                },
                "content": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "required": ["title", "content"],
                            "properties": {
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "children": {"type": ["null", "array"]}
                            }
                        }
                    ]
                },
                "reason": {
                    "type": "string",
                    "minLength": 5,
                    "description": "修改原因说明"
                }
            }
        }


# 直接使用修改操作列表，不需要operations包装
ModificationList = List[ModificationOperation]
