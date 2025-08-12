"""
测试知识树路径验证逻辑
"""

import unittest
from Utils.graph_state import KnowledgeTree, KnowledgeNode
from old_version.knowledge_tree_modifier import KnowledgeTreeModifier

class TestPathValidation(unittest.TestCase):
    def setUp(self):
        """创建测试用的知识树"""
        self.tree = KnowledgeTree(
            root=KnowledgeNode(
                title="根节点",
                content="根内容",
                children=[
                    KnowledgeNode(
                        title="第一章",
                        content="第一章内容",
                        children=[
                            KnowledgeNode(title="第一节", content="第一节内容", children=None),
                            KnowledgeNode(title="第二节", content="第二节内容", children=None)
                        ]
                    ),
                    KnowledgeNode(
                        title="第二章",
                        content="第二章内容",
                        children=None
                    )
                ]
            )
        )
        self.modifier = KnowledgeTreeModifier()

    def test_valid_paths(self):
        """测试有效路径"""
        # 测试获取节点
        node = self.modifier._get_node_by_path(self.tree, ["root", "children", "[0]"])
        self.assertEqual(node.title, "第一章")
        
        node = self.modifier._get_node_by_path(self.tree, ["root", "children", "[0]", "children", "[1]"])
        self.assertEqual(node.title, "第二节")

    def test_invalid_index(self):
        """测试无效索引"""
        with self.assertRaises(ValueError) as cm:
            self.modifier._get_node_by_path(self.tree, ["root", "children", "[5]"])
        self.assertIn("out of range", str(cm.exception))
        self.assertIn("valid range: 0-1", str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            self.modifier._get_node_by_path(self.tree, ["root", "children", "[0]", "children", "[2]"])
        self.assertIn("out of range", str(cm.exception))
        self.assertIn("valid range: 0-1", str(cm.exception))

    def test_negative_index(self):
        """测试负索引"""
        with self.assertRaises(ValueError) as cm:
            self.modifier._get_node_by_path(self.tree, ["root", "children", "[-1]"])
        self.assertIn("out of range", str(cm.exception))

    def test_no_children(self):
        """测试访问没有子节点的节点"""
        with self.assertRaises(ValueError) as cm:
            self.modifier._get_node_by_path(self.tree, ["root", "children", "[1]", "children", "[0]"])
        self.assertIn("Node has no children", str(cm.exception))

    def test_invalid_index_format(self):
        """测试无效索引格式"""
        with self.assertRaises(ValueError) as cm:
            self.modifier._get_node_by_path(self.tree, ["root", "children", "[abc]"])
        self.assertIn("Invalid index format", str(cm.exception))

    def test_modify_operations(self):
        """测试修改操作中的路径验证"""
        from modification_models import ModificationOperation, KnowledgeNodeContent
        
        operations = [
            ModificationOperation(
                action="modify",
                path="root.children[5].content",  # 无效索引
                content=KnowledgeNodeContent(
                    title="无效修改",
                    content="不应该成功",
                    children=None
                ),
                reason="测试无效路径"
            ),
            ModificationOperation(
                action="modify", 
                path="root.children[0].children[2].title",  # 无效索引
                content=KnowledgeNodeContent(
                    title="无效修改",
                    content="不应该成功",
                    children=None
                ),
                reason="测试无效路径"
            )
        ]
        
        modified_tree = self.modifier.modify_knowledge_tree(operations, self.tree)
        stats = self.modifier.get_modification_stats()
        
        self.assertEqual(stats["total_modifications"], 0)
        self.assertEqual(len(stats["errors"]), 2)
        self.assertFalse(stats["success"])

if __name__ == "__main__":
    unittest.main()
