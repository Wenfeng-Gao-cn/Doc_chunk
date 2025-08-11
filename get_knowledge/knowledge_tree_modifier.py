"""
知识树修改模块
根据ModificationList对KnowledgeTree进行增删改操作
"""

import json
from typing import List, Dict, Any, Union, Optional
from copy import deepcopy
from Utils.graph_state import GraphState, KnowledgeTree, KnowledgeNode
from modification_models import ModificationList, ModificationOperation, KnowledgeNodeContent


class KnowledgeTreeModifier:
    """知识树修改器"""
    
    def __init__(self):
        self.modification_count = 0
        self.error_log = []
    
    def modify_knowledge_tree(self, modification_list: ModificationList, 
                            knowledge_tree: KnowledgeTree) -> KnowledgeTree:
        """
        根据ModificationList修改知识树
        
        Args:
            modification_list: ModificationList（现在是List[ModificationOperation]）
            knowledge_tree: 要修改的知识树
            
        Returns:
            修改后的知识树
        """
        # 深拷贝知识树，避免修改原始对象
        modified_tree = deepcopy(knowledge_tree)
        
        # 重置统计
        self.modification_count = 0
        self.error_log = []
        
        # 逐条执行修改指令 - 现在modification_list直接是操作列表
        for i, operation in enumerate(modification_list):
            try:
                if operation.action != 'none':
                    self._execute_action(modified_tree, operation)
                    self.modification_count += 1
            except Exception as e:
                error_msg = f"Operation {i+1} failed: {str(e)}"
                self.error_log.append(error_msg)
                print(f"Warning: {error_msg}")
        
        return modified_tree
    
    def _execute_action(self, tree: KnowledgeTree, operation):
        """执行单个修改操作"""
        action_type = operation.action
        target_path = operation.path
        content = operation.content
        
        if not target_path:
            raise ValueError("Missing path in operation")
        
        if action_type == "add":
            self._add_node(tree, target_path, content)
        elif action_type == "del":
            self._delete_node(tree, target_path)
        elif action_type == "modify":
            self._modify_node(tree, target_path, content)
        else:
            raise ValueError(f"Unknown action: {action_type}")
    
    def _add_node(self, tree: KnowledgeTree, target_path: str, content: KnowledgeNodeContent):
        """添加节点"""
        if content is None:
            raise ValueError("Content cannot be None for add operation")
        
        path_parts = self._parse_path(target_path)
        
        # 如果路径是 "root.children"，则添加到根节点的children
        if len(path_parts) == 2 and path_parts[1] == "children":
            if tree.root.children is None:
                tree.root.children = []
            
            new_node = self._create_node_from_content(content)
            tree.root.children.append(new_node)
        else:
            # 找到父节点并添加
            parent_node = self._get_parent_node(tree, path_parts)
            if parent_node.children is None:
                parent_node.children = []
            
            new_node = self._create_node_from_content(content)
            parent_node.children.append(new_node)
    
    def _delete_node(self, tree: KnowledgeTree, target_path: str):
        """删除节点"""
        path_parts = self._parse_path(target_path)
        
        if len(path_parts) < 3:  # 至少需要 root.children[index]
            raise ValueError("Cannot delete root node")
        
        parent_node, index = self._get_parent_and_index(tree, path_parts)
        
        if parent_node.children and 0 <= index < len(parent_node.children):
            parent_node.children.pop(index)
            if not parent_node.children:
                parent_node.children = None
        else:
            raise ValueError(f"Invalid index {index} for deletion")
    
    def _modify_node(self, tree: KnowledgeTree, target_path: str, content: Any):
        """修改节点"""
        if content is None:
            raise ValueError("Content cannot be None for modify operation")
        
        path_parts = self._parse_path(target_path)
        target_node, attribute = self._get_target_node_and_attribute(tree, path_parts)
        
        if attribute == "title":
            target_node.title = str(content)
        elif attribute == "content":
            target_node.content = str(content)
        elif attribute == "children":
            if isinstance(content, list):
                target_node.children = [self._create_node_from_content(item) for item in content]
            elif content is None:
                target_node.children = None
            else:
                raise ValueError("Children content must be a list or None")
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
    
    def _parse_path(self, path: str) -> List[str]:
        """解析路径字符串"""
        # 将路径分解为组件
        parts = []
        current = ""
        in_brackets = False
        
        for char in path:
            if char == '[':
                if current:
                    parts.append(current)
                    current = ""
                in_brackets = True
            elif char == ']':
                if in_brackets and current:
                    parts.append(f"[{current}]")
                    current = ""
                in_brackets = False
            elif char == '.' and not in_brackets:
                if current:
                    parts.append(current)
                    current = ""
            else:
                current += char
        
        if current:
            parts.append(current)
        
        return parts
    
    def _get_node_by_path(self, tree: KnowledgeTree, path_parts: List[str]) -> KnowledgeNode:
        """根据路径获取节点"""
        current_node = tree.root
        
        for i, part in enumerate(path_parts[1:], 1):  # 跳过 'root'
            if part == "children":
                continue
            elif part.startswith('[') and part.endswith(']'):
                # 处理数组索引
                try:
                    index = int(part[1:-1])
                    if current_node.children is None:
                        raise ValueError(f"Node has no children to access index {index}")
                    if index < 0 or index >= len(current_node.children):
                        raise ValueError(f"Invalid path: index {index} out of range (valid range: 0-{len(current_node.children)-1})")
                    current_node = current_node.children[index]
                except ValueError as e:
                    raise ValueError(f"Invalid index format in path: {part}") from e
            elif part in ["title", "content"]:
                # 这是属性，不是节点
                break
            else:
                # 处理键名访问（如chapter1, section1等）
                if current_node.children is None:
                    raise ValueError(f"Node has no children to access key: {part}")
                
                # 在children列表中查找匹配的节点
                found_node = None
                for child in current_node.children:
                    # 可以根据title匹配或者其他逻辑
                    if (part == "chapter1" and "第一章" in child.title) or \
                       (part == "chapter2" and "第二章" in child.title) or \
                       (part == "chapter3" and "第三章" in child.title) or \
                       (part == "chapter4" and "第四章" in child.title) or \
                       (part == "chapter5" and "第五章" in child.title) or \
                       (part == "chapter6" and "第六章" in child.title) or \
                       (part == "section1" and "第一节" in child.title) or \
                       (part == "section2" and "第二节" in child.title) or \
                       (part == "section3" and "第三节" in child.title) or \
                       (part == "section4" and "第四节" in child.title):
                        found_node = child
                        break
                
                if found_node is None:
                    raise ValueError(f"Cannot find child node for key: {part}")
                
                current_node = found_node
        
        return current_node
    
    def _get_parent_node(self, tree: KnowledgeTree, path_parts: List[str]) -> KnowledgeNode:
        """获取父节点"""
        if len(path_parts) <= 2:
            return tree.root
        
        parent_path = path_parts[:-1]
        return self._get_node_by_path(tree, parent_path)
    
    def _get_parent_and_index(self, tree: KnowledgeTree, path_parts: List[str]) -> tuple:
        """获取父节点和要删除的索引"""
        # 找到包含索引的部分
        for i, part in enumerate(path_parts):
            if part.startswith('[') and part.endswith(']'):
                index = int(part[1:-1])
                parent_path = path_parts[:i]
                parent_node = self._get_node_by_path(tree, parent_path)
                return parent_node, index
        
        raise ValueError("No index found in path for deletion")
    
    def _get_target_node_and_attribute(self, tree: KnowledgeTree, path_parts: List[str]) -> tuple:
        """获取目标节点和属性"""
        if path_parts[-1] in ["title", "content", "children"]:
            attribute = path_parts[-1]
            node_path = path_parts[:-1]
            node = self._get_node_by_path(tree, node_path)
            return node, attribute
        else:
            # 如果没有指定属性，默认修改整个节点
            node = self._get_node_by_path(tree, path_parts)
            return node, "content"
    
    def _create_node_from_content(self, content: Union[KnowledgeNodeContent, dict]) -> KnowledgeNode:
        """从内容创建节点"""
        if isinstance(content, KnowledgeNodeContent):
            # 直接从KnowledgeNodeContent创建
            children = None
            if content.children:
                children = [self._create_node_from_content(child) for child in content.children]
            return KnowledgeNode(title=content.title, content=content.content, children=children)
        elif isinstance(content, dict):
            title = content.get("title", "")
            node_content = content.get("content", "")
            children_data = content.get("children")
            
            children = None
            if children_data:
                if isinstance(children_data, list):
                    children = [self._create_node_from_content(child) for child in children_data]
                elif isinstance(children_data, dict):
                    # 处理字典形式的children
                    children = []
                    for key, value in children_data.items():
                        if isinstance(value, dict):
                            child_node = self._create_node_from_content(value)
                            children.append(child_node)
            
            return KnowledgeNode(title=title, content=node_content, children=children)
        else:
            raise ValueError("Content must be a KnowledgeNodeContent or dictionary for node creation")
    
    def get_modification_stats(self) -> Dict[str, Any]:
        """获取修改统计信息"""
        return {
            "total_modifications": self.modification_count,
            "errors": self.error_log,
            "success": len(self.error_log) == 0
        }


def modify_knowledge_tree(modification_list: ModificationList, 
                         knowledge_tree: KnowledgeTree) -> KnowledgeTree:
    """
    便捷函数：根据ModificationList修改知识树
    
    Args:
        modification_list: ModificationList对象，包含修改指令
        knowledge_tree: 要修改的知识树
        
    Returns:
        修改后的知识树
    """
    modifier = KnowledgeTreeModifier()
    return modifier.modify_knowledge_tree(modification_list, knowledge_tree)


if __name__ == "__main__":
    # 测试代码
    print("开始测试知识树修改模块...")
    
    # 创建测试用的知识树
    test_tree = KnowledgeTree(
        root=KnowledgeNode(
            title="测试根节点",
            content="这是根节点的内容",
            children=[
                KnowledgeNode(
                    title="第一章",
                    content="第一章的内容",
                    children=[
                        KnowledgeNode(
                            title="第一节",
                            content="第一节的内容",
                            children=None
                        )
                    ]
                )
            ]
        )
    )
    
    print("原始知识树结构:")
    print(f"根节点: {test_tree.root.title}")
    if test_tree.root.children:
        print(f"第一个子节点: {test_tree.root.children[0].title}")
        if test_tree.root.children[0].children:
            print(f"第一个孙节点: {test_tree.root.children[0].children[0].title}")
    
    # 测试修改指令
    test_operations = [
        ModificationOperation(
            action="add",
            path="root.children",
            content=KnowledgeNodeContent(
                title="第二章",
                content="第二章的内容",
                children=None
            ),
            reason="添加第二章"
        ),
        ModificationOperation(
            action="modify",
            path="root.children[0].content",
            content=KnowledgeNodeContent(
                title="第一章",
                content="修改后的第一章内容",
                children=None
            ),
            reason="更新第一章内容"
        ),
        ModificationOperation(
            action="add",
            path="root.children[0].children",
            content=KnowledgeNodeContent(
                title="第二节",
                content="第二节的内容",
                children=None
            ),
            reason="在第一章下添加第二节"
        )
    ]
    
    test_modification_list = test_operations  # 现在ModificationList就是List[ModificationOperation]
    
    # 执行修改
    modifier = KnowledgeTreeModifier()
    modified_tree = modifier.modify_knowledge_tree(test_modification_list, test_tree)
    
    print("\n修改后的知识树结构:")
    print(f"根节点: {modified_tree.root.title}")
    print(f"子节点数量: {len(modified_tree.root.children) if modified_tree.root.children else 0}")
    
    if modified_tree.root.children:
        for i, child in enumerate(modified_tree.root.children):
            print(f"第{i+1}个子节点: {child.title}")
            print(f"  内容: {child.content}")
            if child.children:
                for j, grandchild in enumerate(child.children):
                    print(f"  第{j+1}个孙节点: {grandchild.title}")
    
    # 获取修改统计
    stats = modifier.get_modification_stats()
    print(f"\n修改统计:")
    print(f"总修改数: {stats['total_modifications']}")
    print(f"成功: {stats['success']}")
    if stats['errors']:
        print(f"错误: {stats['errors']}")
    
    # 测试便捷函数
    print("\n测试便捷函数...")
    modified_tree2 = modify_knowledge_tree(test_modification_list, test_tree)
    
    print("便捷函数修改测试完成!")
    print("知识树修改模块测试完成！")
