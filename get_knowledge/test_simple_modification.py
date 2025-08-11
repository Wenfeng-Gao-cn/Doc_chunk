"""
简单测试修改后的模型
"""

from modification_models import ModificationList, ModificationOperation, KnowledgeNodeContent

def test_modification_models():
    print("=== 测试修改后的Pydantic模型 ===")
    
    # 测试创建ModificationOperation
    try:
        op1 = ModificationOperation(
            action="modify",
            path="root.children[0]",
            content=KnowledgeNodeContent(
                title="测试标题",
                content="测试内容",
                children=None
            ),
            reason="测试修改操作"
        )
        print(f"✓ 成功创建ModificationOperation: {op1.action}")
        
        # 测试创建ModificationList（现在是List[ModificationOperation]）
        modification_list = [op1]
        print(f"✓ 成功创建ModificationList，包含 {len(modification_list)} 个操作")
        
        # 测试JSON序列化
        import json
        json_data = [op.dict() for op in modification_list]
        print(f"✓ 成功序列化为JSON: {len(json_data)} 个操作")
        
        # 测试从JSON反序列化
        reconstructed_ops = [ModificationOperation(**op_dict) for op_dict in json_data]
        print(f"✓ 成功从JSON反序列化: {len(reconstructed_ops)} 个操作")
        
        print("\n=== 所有测试通过！ ===")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

if __name__ == "__main__":
    test_modification_models()
