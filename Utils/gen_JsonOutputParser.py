from pydantic import BaseModel, Field
from typing import Type, Dict, Any, Set, List, Optional, Union, get_origin, get_args
import json
import re
import inspect
from Utils.logger import setup_logger

# 初始化logger
logger = setup_logger(__name__)

def gen_JsonOutputParser(prompt: str, model: Type[BaseModel]) -> str:
    """
    创建一个带有格式化输出要求的提示词，包括详细的字段描述和示例。
    
    Args:
        prompt: 原始提示词，不包含输出格式的定义部分。
        model: Pydantic模型，定义了输出的格式。
        
    Returns:
        完整的提示词，包括输出格式的要求和字段描述。
    """
    # 获取模型的 JSON schema
    schema = model.model_json_schema()
    
    # 创建一个更丰富的示例实例
    example = create_example_from_schema(schema, schema, set(), depth=0)
    
    # 将示例转换为 JSON 字符串，使用更好的格式
    example_json_str = json.dumps(example, indent=2, ensure_ascii=False)
    
    # 转义JSON字符串中的花括号，防止与format()冲突
    escaped_json_str = example_json_str.replace("{", "{{").replace("}", "}}")
    
    # 生成字段描述文本
    descriptions = get_field_descriptions(model)
    descriptions_text = ""
    if descriptions:
        descriptions_text = "\n### 字段说明：\n" + "\n".join(descriptions) + "\n"
    
    # 生成模型结构说明
    structure_info = get_model_structure_info(model)
    
    # 生成完整的输出格式说明 - 使用转义后的JSON
    format_instruction = """
## 输出格式要求

请严格按照以下JSON格式输出您的回答：

```json
""" + escaped_json_str + """
```
""" + descriptions_text + """
### 结构说明：
""" + structure_info + """

### 重要提醒：
1. 请确保您的回答是有效的JSON格式
2. 必须包含所有必需的字段
3. 字段类型必须与上述规范一致
4. 对于数组类型，请提供完整的元素
5. 对于嵌套对象，请确保结构完整
6. 不要添加任何JSON格式之外的额外文本或解释
"""
    
    # 组合原始提示词和格式说明
    full_prompt = prompt.strip() + format_instruction
    
    return full_prompt

def get_model_structure_info(model: Type[BaseModel], level: int = 0) -> str:
    """
    生成模型结构的详细说明
    
    Args:
        model: Pydantic模型
        level: 嵌套层级
        
    Returns:
        结构说明字符串
    """
    indent = "  " * level
    info_lines = []
    
    model_name = model.__name__
    info_lines.append(f"{indent}- {model_name} (对象类型)")
    
    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        required = "必需" if field_info.is_required() else "可选"
        
        # 处理不同的字段类型
        type_desc = get_field_type_description(field_type)
        info_lines.append(f"{indent}  · {field_name}: {type_desc} ({required})")
        
        # 如果是嵌套的BaseModel，递归显示结构
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            nested_info = get_model_structure_info(field_type, level + 2)
            info_lines.append(nested_info)
        elif get_origin(field_type) is list:
            args = get_args(field_type)
            if args and inspect.isclass(args[0]) and issubclass(args[0], BaseModel):
                info_lines.append(f"{indent}    数组元素结构:")
                nested_info = get_model_structure_info(args[0], level + 3)
                info_lines.append(nested_info)
    
    return "\n".join(info_lines)

def get_field_type_description(field_type: Any) -> str:
    """
    获取字段类型的友好描述
    
    Args:
        field_type: 字段类型
        
    Returns:
        类型描述字符串
    """
    if field_type is str:
        return "字符串"
    elif field_type is int:
        return "整数"
    elif field_type is float:
        return "浮点数"
    elif field_type is bool:
        return "布尔值"
    elif get_origin(field_type) is list:
        args = get_args(field_type)
        if args:
            item_type_desc = get_field_type_description(args[0])
            return f"{item_type_desc}数组"
        return "数组"
    elif get_origin(field_type) is dict:
        return "字典对象"
    elif get_origin(field_type) is Union:
        args = get_args(field_type)
        # 处理Optional类型 (Union[X, None])
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return f"可选的{get_field_type_description(non_none_type)}"
        else:
            type_descs = [get_field_type_description(arg) for arg in args]
            return f"联合类型({' | '.join(type_descs)})"
    elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
        return f"{field_type.__name__}对象"
    else:
        return str(field_type).replace('typing.', '')

def get_field_descriptions(model: Type[BaseModel], prefix: str = "", result: List[str] = None) -> List[str]:
    """
    递归获取模型及其嵌套模型中所有字段的描述。
    
    Args:
        model: Pydantic模型。
        prefix: 当前字段的前缀路径。
        result: 已收集的描述列表。
        
    Returns:
        字段描述列表。
    """
    if result is None:
        result = []
    
    # 获取模型的字段信息
    for field_name, field_info in model.model_fields.items():
        field_type = field_info.annotation
        field_path = f"{prefix}.{field_name}" if prefix else field_name
        
        # 构建字段描述
        description_parts = []
        if field_info.description:
            description_parts.append(field_info.description)
        
        # 添加类型信息
        type_desc = get_field_type_description(field_type)
        description_parts.append(f"类型: {type_desc}")
        
        # 添加必需性信息
        required_desc = "必需" if field_info.is_required() else "可选"
        description_parts.append(f"{required_desc}")
        
        # 组合完整描述
        full_description = " | ".join(description_parts)
        result.append(f"- **{field_path}**: {full_description}")
        
        # 如果字段类型是另一个Pydantic模型，递归处理
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            get_field_descriptions(field_type, field_path, result)
        
        # 处理Optional类型
        elif get_origin(field_type) is Union:
            args = get_args(field_type)
            if len(args) == 2 and type(None) in args:
                non_none_type = args[0] if args[1] is type(None) else args[1]
                if inspect.isclass(non_none_type) and issubclass(non_none_type, BaseModel):
                    get_field_descriptions(non_none_type, field_path, result)
        
        # 处理列表类型，检查列表项是否为Pydantic模型
        elif get_origin(field_type) is list and get_args(field_type):
            item_type = get_args(field_type)[0]
            if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
                get_field_descriptions(item_type, f"{field_path}[items]", result)
        
        # 处理字典类型，检查值是否为Pydantic模型
        elif get_origin(field_type) is dict and len(get_args(field_type)) >= 2:
            value_type = get_args(field_type)[1]
            if inspect.isclass(value_type) and issubclass(value_type, BaseModel):
                get_field_descriptions(value_type, f"{field_path}[values]", result)
    
    return result

def create_example_from_schema(schema: Dict[str, Any], root_schema: Dict[str, Any], 
                             visited_refs: Set[str], depth: int = 0, max_depth: int = 3) -> Any:
    """
    从 JSON schema 创建更丰富的示例。
    
    Args:
        schema: 当前处理的 JSON schema。
        root_schema: 根 JSON schema，用于解析引用。
        visited_refs: 已经访问过的引用，用于检测循环引用。
        depth: 当前递归深度。
        max_depth: 最大递归深度，防止无限递归。
        
    Returns:
        示例值。
    """
    # 防止过深递归
    if depth > max_depth:
        return "..."
    
    # 如果 schema 是空的或者不是字典，返回 None
    if not schema or not isinstance(schema, dict):
        return None
    
    # 如果 schema 中有 example，直接使用
    if "example" in schema:
        return schema["example"]
    
    # 处理引用
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in visited_refs:
            # 检测到循环引用，返回简化版本
            return {"...": "循环引用"}
        
        visited_refs.add(ref)
        try:
            if ref.startswith("#/$defs/") or ref.startswith("#/definitions/"):
                definition_name = ref.split("/")[-1]
                definitions = root_schema.get("$defs", root_schema.get("definitions", {}))
                if definition_name in definitions:
                    result = create_example_from_schema(definitions[definition_name], root_schema, 
                                                      visited_refs.copy(), depth + 1, max_depth)
                    visited_refs.discard(ref)
                    return result
        finally:
            visited_refs.discard(ref)
    
    # 获取 schema 的类型
    schema_type = schema.get("type", "object")
    
    # 处理不同类型的 schema
    if schema_type == "string":
        # 根据字段名生成更有意义的示例
        title = schema.get("title", "").lower()
        if "title" in title or "name" in title:
            return "示例标题"
        elif "content" in title:
            return "这里是详细的内容描述"
        elif "id" in title:
            return "unique_id_123"
        elif "url" in title or "link" in title:
            return "https://example.com"
        elif "email" in title:
            return "example@email.com"
        else:
            return "示例字符串值"
    elif schema_type == "integer":
        return 42
    elif schema_type == "number":
        return 3.14
    elif schema_type == "boolean":
        return True
    elif schema_type == "array":
        items_schema = schema.get("items", {})
        # 为数组创建多个示例项
        example_item = create_example_from_schema(items_schema, root_schema, 
                                                visited_refs.copy(), depth + 1, max_depth)
        # 对于简单类型，创建多个示例
        if isinstance(example_item, (str, int, float, bool)):
            return [example_item, f"{example_item}_2"] if isinstance(example_item, str) else [example_item, example_item + 1]
        else:
            return [example_item]
    elif schema_type == "object":
        result = {}
        properties = schema.get("properties", {})
        
        for prop_name, prop_schema in properties.items():
            result[prop_name] = create_example_from_schema(prop_schema, root_schema, 
                                                         visited_refs.copy(), depth + 1, max_depth)
        
        # 处理字典类型（additionalProperties）
        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            additional_example = create_example_from_schema(
                schema["additionalProperties"], 
                root_schema, 
                visited_refs.copy(), 
                depth + 1, 
                max_depth
            )
            if additional_example is not None:
                result["additional_property"] = additional_example
        
        return result
    
    # 处理 oneOf、anyOf、allOf
    for key in ["oneOf", "anyOf", "allOf"]:
        if key in schema and schema[key]:
            # 选择第一个可行的模式
            for subschema in schema[key]:
                example = create_example_from_schema(subschema, root_schema, 
                                                   visited_refs.copy(), depth + 1, max_depth)
                if example is not None:
                    return example
    
    # 默认返回空对象
    return {}

if __name__ == "__main__":
    # 改进的KnowledgeTree模型定义
    class KnowledgeNode(BaseModel):
        """表示单个知识点节点的模型"""
        title: str = Field(..., description="知识点的标题，应该简洁明确地概括该知识点的核心内容")
        content: str = Field(..., description="知识点的详细内容说明，包含具体的解释、要点或示例")
        level: int = Field(default=1, description="知识点的层级深度，根节点为1，子节点依次递增")
        tags: Optional[List[str]] = Field(
            default=None, 
            description="知识点的标签列表，用于分类和检索，如['基础概念', '重要', '理论']"
        )
        children: Optional[List['KnowledgeNode']] = Field(
            default=None, 
            description="子知识点列表，包含该知识点下的所有子节点，如果没有子节点则为null"
        )
        
        class Config:
            # 允许前向引用
            arbitrary_types_allowed = True

    class KnowledgeMetadata(BaseModel):
        """知识树的元数据信息"""
        total_nodes: int = Field(..., description="知识树中总的节点数量")
        max_depth: int = Field(..., description="知识树的最大深度")
        created_date: str = Field(..., description="知识树创建日期，格式为YYYY-MM-DD")
        subject: str = Field(..., description="知识树所属的学科或主题领域")

    class KnowledgeTree(BaseModel):
        """表示完整知识点树的模型"""
        metadata: KnowledgeMetadata = Field(..., description="知识树的元数据信息")
        root: KnowledgeNode = Field(..., description="知识点树的根节点，包含整个知识体系的起始点")
        summary: str = Field(..., description="整个知识树的概要说明，简述其覆盖的主要内容和学习目标")
        
        class Config:
            # 允许前向引用
            arbitrary_types_allowed = True

    # 更新模型引用
    KnowledgeNode.model_rebuild()
    KnowledgeTree.model_rebuild()
    
    # 测试提示词模板
    prompt_template = """
你是一位专业的教育内容专家。请根据用户提供的主题创建一个结构化的知识点树。

# 任务要求
- 主题: {topic}
- 目标学习者: {target_audience}
- 期望深度: {depth_level}

# 创建指导原则
1. 知识点应该层次分明，逻辑清晰
2. 每个节点都应该有实用的内容描述
3. 使用适当的标签帮助分类
4. 确保知识点之间的关联性
5. 适合目标学习者的认知水平

请创建一个完整的知识点树结构。
"""
    
    # 生成完整提示词
    formatted_prompt = gen_JsonOutputParser(prompt_template, KnowledgeTree)
    
    # 测试输入参数
    test_input = {
        "topic": "Python编程基础",
        "target_audience": "编程初学者",
        "depth_level": "入门到进阶"
    }
    
    # 打印结果
    print("="*100)
    print("优化后的完整提示词：")
    print("="*100)
    final_prompt = formatted_prompt.format(**test_input)
    print(final_prompt)
    print("="*100)
    
    # 详细验证
    print("\n详细验证结果：")
    print("-"*80)
    
    # 1. 检查JSON示例的完整性
    json_start = final_prompt.find('```json')
    json_end = final_prompt.find('```', json_start + 7)
    if json_start != -1 and json_end != -1:
        json_content = final_prompt[json_start+7:json_end].strip()
        try:
            parsed_json = json.loads(json_content)
            print("✓ JSON示例语法正确")
            print(f"✓ JSON包含字段: {list(parsed_json.keys())}")
            
            # 检查嵌套结构
            if 'root' in parsed_json and 'children' in parsed_json['root']:
                print("✓ 包含嵌套的children结构")
            if 'metadata' in parsed_json:
                print("✓ 包含metadata元数据")
                
        except json.JSONDecodeError as e:
            print(f"✗ JSON语法错误: {e}")
    else:
        print("✗ 未找到JSON示例")
    
    # 2. 检查字段描述
    field_descriptions = [
        "title", "content", "level", "tags", "children", 
        "metadata", "root", "summary", "total_nodes", "max_depth"
    ]
    found_descriptions = sum(1 for field in field_descriptions if field in final_prompt)
    print(f"✓ 字段描述覆盖率: {found_descriptions}/{len(field_descriptions)} ({found_descriptions/len(field_descriptions)*100:.1f}%)")
    
    # 3. 检查结构说明
    structure_indicators = ["结构说明", "必需", "可选", "对象类型", "数组元素结构"]
    found_structure = sum(1 for indicator in structure_indicators if indicator in final_prompt)
    print(f"✓ 结构说明完整性: {found_structure}/{len(structure_indicators)} 个关键指标")
    
    # 4. 检查提示词组织
    key_sections = ["输出格式要求", "字段说明", "重要提醒"]
    found_sections = sum(1 for section in key_sections if section in final_prompt)
    print(f"✓ 提示词章节完整性: {found_sections}/{len(key_sections)}")
    
    # 5. 长度和复杂度
    print(f"✓ 最终提示词长度: {len(final_prompt)} 字符")
    print(f"✓ 输入参数正确注入: {'Python编程基础' in final_prompt}")
    
    print("-"*80)
    
    # 展示关键改进点
    print("\n关键改进点：")
    print("1. 🔧 增强的类型描述：支持Optional、Union等复杂类型")
    print("2. 📊 丰富的JSON示例：基于字段名生成有意义的示例值")
    print("3. 🏗️ 详细的结构说明：清晰展示嵌套关系和层级结构")
    print("4. 🎯 改进的模型设计：添加了metadata、tags等实用字段")
    print("5. 🛡️ 循环引用保护：防止无限递归，提供更稳定的示例生成")
    print("6. 📝 更好的文档格式：使用markdown格式提升可读性")