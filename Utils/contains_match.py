import re
import string

class FormatInsensitiveMatcher:
    def __init__(self):
        # 定义需忽略的字符：空白符 + 中英文标点 + Markdown 符号
        self.ignore_chars = set(
            string.whitespace + 
            string.punctuation + 
            '、。！？，；：“”‘’（）【】《》……—·＇＂＃＄％＆＇＊＋－／：＜＝＞＠［＼］＾＿｀｛｜｝～' +
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