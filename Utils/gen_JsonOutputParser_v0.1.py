from pydantic import BaseModel, Field
from typing import Type, Dict, Any, Set, List, get_origin, get_args
import json
import re
import inspect
from Utils.logger import setup_logger

# 初始化logger
logger = setup_logger(__name__)

def gen_JsonOutputParser(prompt: str, model: Type[BaseModel]) -> str:
    """
    创建一个带有格式化输出要求的提示词，包括字段描述。
    
    Args:
        model: Pydantic模型，定义了输出的格式。
        prompt: 原始提示词，不包含输出格式的定义部分。
        
    Returns:
        完整的提示词，包括输出格式的要求和字段描述。
    """
    # 获取模型的 JSON schema
    schema = model.model_json_schema()
    
    # 创建一个示例实例
    example = create_example_from_schema(schema, schema, set())
    
    # 将示例转换为 JSON 字符串
    example_json_str = json.dumps(example, indent=2, ensure_ascii=False)
    
    # 安全转义花括号（只转义最外层花括号）
    escaped_json_str = re.sub(r'(?<!\{)\{(?!\{)', '{{', example_json_str)
    escaped_json_str = re.sub(r'(?<!\})\}(?!\})', '}}', escaped_json_str)
    
    # 生成字段描述文本
    descriptions = get_field_descriptions(model)
    descriptions_text = ""
    if descriptions:
        descriptions_text = "字段说明：\n" + "\n".join(descriptions) + "\n\n"
    
    # 生成描述模型的文本
    model_description = "请使用以下JSON格式输出您的回答：\n\n"
    model_description += "```json\n" + escaped_json_str + "\n```\n\n"
    if descriptions_text:
        model_description += descriptions_text
    model_description += "请确保您的回答是有效的JSON，并且遵循上述结构。不要添加任何额外的文本或解释。"
    
    # 组合原始提示词和模型描述
    full_prompt = prompt + "\n\n" + model_description
    
    return full_prompt

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
        
        # 获取字段描述
        if field_info.description:
            result.append(f"- {field_path}: {field_info.description}")
        
        # 如果字段类型是另一个Pydantic模型，递归处理
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            get_field_descriptions(field_type, field_path, result)
        
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

def create_example_from_schema(schema: Dict[str, Any], root_schema: Dict[str, Any], visited_refs: Set[str]) -> Any:
    """
    从 JSON schema 创建示例。
    
    Args:
        schema: 当前处理的 JSON schema。
        root_schema: 根 JSON schema，用于解析引用。
        visited_refs: 已经访问过的引用，用于检测循环引用。
        
    Returns:
        示例值。
    """
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
            # 检测到循环引用
            return {}
        visited_refs.add(ref)
        if ref.startswith("#/$defs/") or ref.startswith("#/definitions/"):
            definition_name = ref.split("/")[-1]
            definitions = root_schema.get("$defs", root_schema.get("definitions", {}))
            if definition_name in definitions:
                return create_example_from_schema(definitions[definition_name], root_schema, visited_refs)
    
    # 获取 schema 的类型
    schema_type = schema.get("type", "object")
    
    # 处理不同类型的 schema
    if schema_type == "string":
        return "示例字符串"
    elif schema_type == "integer":
        return 0
    elif schema_type == "number":
        return 0.0
    elif schema_type == "boolean":
        return False
    elif schema_type == "array":
        items_schema = schema.get("items", {})
        return [create_example_from_schema(items_schema, root_schema, visited_refs.copy())]
    elif schema_type == "object":
        result = {}
        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            result[prop_name] = create_example_from_schema(prop_schema, root_schema, visited_refs.copy())
        
        # 处理字典类型（additionalProperties）
        if "additionalProperties" in schema:
            result["示例键"] = create_example_from_schema(
                schema["additionalProperties"], 
                root_schema, 
                visited_refs.copy()
            )
        return result
    
    # 处理 oneOf、anyOf、allOf
    for key in ["oneOf", "anyOf", "allOf"]:
        if key in schema and schema[key]:
            # 选择第一个可行的模式
            for subschema in schema[key]:
                example = create_example_from_schema(subschema, root_schema, visited_refs.copy())
                if example is not None:
                    return example
    
    # 默认返回空对象
    return {}

if __name__ == "__main__":
    # 1. 定义与ThinkingNode完全匹配的Pydantic模型
    class ThinkingStep(BaseModel):
        evaluation_previous_progress: str = Field(
            description="对先前进度的评估，可能的值为'成功'、'失败'或'未知'，后跟简要分析当前进度，检查先前的思考是否朝着正确的方向发展的原因"
        )
        important_contents: str = Field(
            description="与用户指令密切相关的重要内容。如果有，请输出内容；如果没有，请输出空字符串 ''"
        )
        thought: str = Field(
            description="详细的思考过程，包含已经完成的要求和下一步需要完成的要求。如果evaluation_previous_progress是'失败'，请在此输出反思内容"
        )
        next_goal: str = Field(
            description="根据思考过程生成的下一步行动的简短自然语言描述目标"
        )
    
    # 2. 使用ThinkingNode中的实际提示模板
    prompt_template = """
你是一个设计用于解决复杂任务的AI助手。你的目标是通过深入思考，逐步分解并完成用户的要求。

# 输入格式
- 任务描述: {task}
- 先前的思考步骤: {previous_thinking}
- 当前进度: {current_progress}
- 待解决的问题: {pending_issues}

2. 思考过程：
- 将复杂任务分解为更小的子任务
- 一次专注于一个子任务
- 确保每个子任务都朝着总体目标前进
- 保持逻辑清晰，推理连贯
- 如果遇到困难，尝试不同的方法
- 利用已有的知识和信息
- 在必要时明确需要哪些额外信息

3. 任务完成：
- 当你认为已经完成了所有子任务，请进行全面的自我检查
- 确保所有用户的要求都已得到满足
- 如果有任何遗漏，请返回并完成
- 在最终的思考中，总结整个过程和结果

4. 记忆和上下文：
- 始终跟踪已完成的步骤和剩余的步骤
- 保持对整体任务的关注
- 引用先前的思考以保持连贯性
- 不要重复已经完成的工作
"""
    
    # 3. 调用函数生成完整提示词
    formatted_prompt = gen_JsonOutputParser(prompt_template, ThinkingStep)
    
    # 4. 添加模拟输入参数进行验证
    test_input = {
        "task": "分析2023年全球科技行业的发展趋势",
        "previous_thinking": "已确定分析框架",
        "current_progress": "开始收集AI领域数据",
        "pending_issues": ["需要更多可再生能源领域数据", "需要整合交叉影响"]
    }
    
    # 5. 打印完整结果（包含参数占位符）
    print("="*80)
    print("生成的完整提示词：")
    print("="*80)
    print(formatted_prompt.format(**test_input))
    print("="*80)
    
    # 6. 添加验证步骤
    print("\n验证关键元素：")
    print("-"*60)
    print("1. JSON示例存在:", "```json" in formatted_prompt)
    print("2. 字段描述存在:", any(field in formatted_prompt for field in [
        "evaluation_previous_progress", 
        "important_contents", 
        "thought", 
        "next_goal"
    ]))
    print("3. 输入参数被正确注入:", "分析2023年全球科技行业的发展趋势" in formatted_prompt.format(**test_input))
    print("4. 结构完整性检查:", formatted_prompt.count("{") == formatted_prompt.count("}"))
    print("-"*60)
