"""
测试简化的JSON处理方式
"""

import json
import asyncio
from typing import Dict, List, Any


def test_simple_json_structure():
    """测试简化的JSON结构"""
    
    # 模拟LLM输出的简化JSON结构
    sample_output = {
        "modifications": [
            {
                "action": "add",
                "path": "root.children.chapter1.children",
                "content": {
                    "title": "第七条",
                    "content": "国家对电信业务经营实行许可制度。",
                    "children": None
                },
                "reason": "补充遗漏的第七条内容"
            },
            {
                "action": "modify",
                "path": "root.children.chapter2.children.section1.children.article8",
                "content": {
                    "title": "第八条",
                    "content": "电信业务分为基础电信业务和增值电信业务。",
                    "children": None
                },
                "reason": "修正第八条内容不完整"
            },
            {
                "action": "del",
                "path": "root.children.chapter6.children.article99",
                "content": None,
                "reason": "删除不存在的条款"
            },
            {
                "action": "none",
                "path": "",
                "content": None,
                "reason": "无需修改"
            }
        ]
    }
    
    print("=== 简化JSON结构测试 ===")
    print(json.dumps(sample_output, indent=2, ensure_ascii=False))
    
    # 处理修改操作
    modifications = sample_output.get('modifications', [])
    active_modifications = [m for m in modifications if m.get('action') != 'none']
    
    print(f"\n总修改操作数: {len(modifications)}")
    print(f"有效修改操作数: {len(active_modifications)}")
    
    for i, mod in enumerate(active_modifications):
        print(f"\n修改 {i+1}:")
        print(f"  操作类型: {mod['action']}")
        print(f"  目标路径: {mod['path']}")
        print(f"  修改原因: {mod['reason']}")
        if mod['content']:
            print(f"  内容标题: {mod['content']['title']}")
            print(f"  内容摘要: {mod['content']['content'][:50]}...")


def compare_with_original_structure():
    """对比原始复杂结构"""
    
    # 原始复杂结构（从您的输出中提取）
    original_complex = {
        'operations': [
            {
                'is_modification_needed': True,
                'modification_action': 'add',
                'target_path': 'root.children.chapter2.children.section1.children',
                'modification_content': {
                    'title': '第七条第三款', 
                    'content': '未取得电信业务经营许可证，任何组织或者个人不得从事电信业务经营活动。', 
                    'children': None
                },
                'reason': '原文第七条第三款内容未在知识树中体现'
            }
        ]
    }
    
    # 简化结构
    simplified = {
        "modifications": [
            {
                "action": "add",
                "path": "root.children.chapter2.children.section1.children",
                "content": {
                    "title": "第七条第三款",
                    "content": "未取得电信业务经营许可证，任何组织或者个人不得从事电信业务经营活动。",
                    "children": None
                },
                "reason": "原文第七条第三款内容未在知识树中体现"
            }
        ]
    }
    
    print("\n=== 结构对比 ===")
    print("\n原始复杂结构:")
    print(json.dumps(original_complex, indent=2, ensure_ascii=False))
    
    print("\n简化结构:")
    print(json.dumps(simplified, indent=2, ensure_ascii=False))
    
    print("\n优势对比:")
    print("1. 字段数量: 复杂结构5个字段 vs 简化结构4个字段")
    print("2. 嵌套层级: 减少了一层嵌套")
    print("3. 字段命名: 更简洁直观")
    print("4. 处理逻辑: 更简单的条件判断")


def test_json_processing_performance():
    """测试JSON处理性能"""
    import time
    
    # 生成大量测试数据
    large_data = {
        "modifications": []
    }
    
    for i in range(1000):
        large_data["modifications"].append({
            "action": "add" if i % 3 == 0 else "modify",
            "path": f"root.children.chapter{i//10}.children.article{i}",
            "content": {
                "title": f"第{i}条",
                "content": f"这是第{i}条的内容，包含一些测试文本。" * 5,
                "children": None
            },
            "reason": f"修改原因{i}"
        })
    
    print("\n=== 性能测试 ===")
    
    # 测试JSON序列化
    start_time = time.time()
    json_str = json.dumps(large_data, ensure_ascii=False)
    serialize_time = time.time() - start_time
    
    # 测试JSON反序列化
    start_time = time.time()
    parsed_data = json.loads(json_str)
    deserialize_time = time.time() - start_time
    
    # 测试数据处理
    start_time = time.time()
    active_mods = [m for m in parsed_data["modifications"] if m["action"] != "none"]
    process_time = time.time() - start_time
    
    print(f"数据量: {len(large_data['modifications'])} 个修改操作")
    print(f"JSON序列化时间: {serialize_time:.4f}秒")
    print(f"JSON反序列化时间: {deserialize_time:.4f}秒")
    print(f"数据处理时间: {process_time:.4f}秒")
    print(f"有效修改操作: {len(active_mods)}")


if __name__ == "__main__":
    test_simple_json_structure()
    compare_with_original_structure()
    test_json_processing_performance()
