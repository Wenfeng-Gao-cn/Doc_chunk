"""
修改操作模型定义
包含ModificationOperation和ModificationList的定义
"""

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Any, List


class ModificationOperation(BaseModel):
    """单个修改操作模型"""
    is_modification_needed: bool = Field(
        ...,
        description="标识是否需要修改操作"
    )
    
    modification_action: Optional[Literal['add', 'del', 'modify']] = Field(
        None,
        description="修改类型：添加(add)/删除(del)/修改(modify)"
    )
    
    target_path: str = Field(
        ...,
        min_length=1,
        description="目标节点路径，如'root.children'或'root.children.chapter6.children'"
    )
    
    modification_content: Optional[Any] = Field(
        None,
        description="修改内容对象，操作类型为del时可为null"
    )
    
    reason: str = Field(
        ...,
        min_length=5,
        max_length=100,
        description="修改原因的简要说明"
    )

    # 业务逻辑验证器
    @model_validator(mode='after')
    def validate_operation(self):
        if not self.is_modification_needed:
            if self.modification_action or self.modification_content:
                raise ValueError("不需要修改时不应指定修改类型或修改内容")
            return self
            
        if not self.modification_action:
            raise ValueError("需要修改时必须指定修改类型")
            
        if self.modification_action in ['add', 'modify']:
            if self.modification_content is None:
                raise ValueError(f"{self.modification_action}操作必须提供内容")
        else:  # del 操作
            if self.modification_content is not None:
                raise ValueError("删除操作不应提供修改内容")
                
        return self


class ModificationList(BaseModel):
    """顶层修改操作列表模型"""
    operations: List[ModificationOperation] = Field(
        ...,
        description="修改操作集合",
        min_length=1
    )