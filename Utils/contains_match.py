import re
import string
from pydantic import BaseModel, Field
import yaml
from Utils.llm import get_llm
from Utils.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts  import PromptTemplate
from rich import print

logger = setup_logger(__name__)

class llm_match_result(BaseModel):
    
    reason: str = Field(
        ...,
        description="错误分析，评估理由，修改操作计划"
    )
    found: bool = Field(True, description="验证结果：切片内容是否与原文匹配，如果切片内容存在于原文中返回true，否则返回false")

class llm_recorrect(BaseModel):
    corrected_content: str = Field(
        ...,
        description="更正后的，正确的内容"
    )

class llm_eva_result(BaseModel):
    reason: str
    found: bool
    corrected_content: str    

class FormatInsensitiveMatcher:
    def __init__(self):
        # 定义需忽略的字符：空白符 + 中英文标点 + Markdown 符号
        self.ignore_chars = set(
            string.whitespace + 
            string.punctuation + 
            '、。！？，；：“”‘’（）【】《》……—·＇＂＃＄％＆＇＊＋－／：＜＝＞＠［＼］＾＿｀｛｜｝～——' +
            '*_`#'  # Markdown 符号
        )
        # 预编译正则表达式（包含大小写忽略）
        self.regex = re.compile(f"[{re.escape(''.join(self.ignore_chars))}]", re.IGNORECASE)
    
    def clean_text(self, text):
        """移除所有特殊字符并统一为小写格式"""
        return self.regex.sub('', text).casefold()
    
    def contains_match(self, target, document):
        """
        检查目标字符串是否存在于文档中（忽略格式和大小写）
        参数:
            target: 要查找的目标字符串 (str)
            document: 被搜索的原始文档 (str)
        返回:
            bool: 存在返回 True，否则 False
        """
        clean_target = self.clean_text(target)
        # 空目标始终返回 True
        if not clean_target:
            return True
        clean_doc = self.clean_text(document)
        return clean_target in clean_doc

    async def _match_llm_chain(self,chunk_title, target, document) -> llm_eva_result: 
        setup_file = "setup.yaml"
        try:
            with open(setup_file, encoding='utf-8') as f:
                config = yaml.safe_load(f)
                llm_match_prompt_file = config["graph_config"]["llm_matcher_prompt"]
                llm_recorrect_prompt_file = config["graph_config"]["llm_recorrect_prompt"]
                try:
                    with open(llm_match_prompt_file, encoding='utf-8') as prompt_file:
                        llm_match_prompt = prompt_file.read()
                    with open(llm_recorrect_prompt_file,encoding='utf-8') as prompt_file:
                        self.llm_recorrect_prompt = prompt_file.read()
                except FileNotFoundError:
                    logger.error(f"Prompt file not found: {llm_match_prompt_file}")
                    raise
                except Exception as e:
                    logger.error(f"Error reading prompt file: {e}")
                    raise
        except yaml.YAMLError as ye:
            logger.error(f"加载{setup_file}配置失败: {str(ye)}", exc_info=True)
            raise
        except KeyError as ke:
            logger.error(f"缺少配置键: {str(ke)} - 文件: {setup_file}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"加载{setup_file}配置失败: {str(e)}", exc_info=True)
            raise
        parser = JsonOutputParser(pydantic_object=llm_match_result)
        format_instructions = parser.get_format_instructions()
        input_2_llm = PromptTemplate(
            input_variables=["chunk_title","target", "document"],
            template=llm_match_prompt,
            partial_variables={"format_instructions": format_instructions}
        )
        chain = input_2_llm | get_llm("matcher_llm") | parser
        result_dict = chain.invoke({
            "chunk_title": chunk_title,
            "target": target,
            "document": document
        })
        result = llm_match_result(**result_dict)
        eval_result = llm_eva_result(
            reason=result.reason,
            found=result.found,
            corrected_content=""
        )
        
        if not result.found:
            print(">>>通过大模型进行内容修复>>>")
            correct_result = await self._recorrect_llm_chain(chunk_title,target, result.reason, document)
            eval_result.corrected_content = correct_result.corrected_content

        return eval_result

    async def _recorrect_llm_chain(self, chunk_title,target, reason, document) -> llm_recorrect:
        parser = JsonOutputParser(pydantic_object=llm_eva_result)
        format_instructions = parser.get_format_instructions()
        input_2_llm = PromptTemplate(
            input_variables=["chunk_title","target","reason","document"],
            template=self.llm_recorrect_prompt,
            partial_variables={"format_instructions": format_instructions}
        )         
        chain = input_2_llm | get_llm("matcher_llm") | parser
        result_dict = chain.invoke({
            "chunk_title": chunk_title,
            "target": target,
            "reason": reason,
            "document": document
        })
        result = llm_recorrect(**result_dict)
        return result

    async def contains_match_llm(self, chunk_title,target, document) -> llm_eva_result:
        """使用LLM检查目标内容是否匹配文档，返回评估结果对象"""
        return await self._match_llm_chain(chunk_title,target, document)

# 使用示例
if __name__ == "__main__":
    matcher = FormatInsensitiveMatcher()
    
    # 测试用例
    test_cases = [
        ("HELLO", "**Hello** World!", True),     # 忽略 Markdown 和大小写
        ("Py_Thon", "Py*thon\nCode", True),      # 忽略符号和换行
        ("测试", "【测试】案例！", True),         # 忽略中文标点
        ("NO_MATCH", "Text without match", False),
        ("", "Any document", True)               # 空目标处理
    ]
    
    for target, doc, expected in test_cases:
        result = matcher.contains_match(target, doc)
        print(f"目标: '{target}' | 文档: '{doc}' | 匹配: {result} (预期: {expected})")
