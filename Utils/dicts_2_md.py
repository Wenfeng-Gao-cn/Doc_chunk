from typing import List, Any
import json
from datetime import datetime

class DocumentDisplayFormatter:
    """用于格式化和显示LangChain Document对象的工具类"""
    
    def __init__(self, max_content_length: int = 300):
        self.max_content_length = max_content_length
    
    def to_markdown(self, documents: List[Any]) -> str:
        """将Document列表转换为Markdown格式"""
        if not documents:
            return "**No documents to display**"
        
        markdown_parts = [f"#### 引用了【{len(documents)} 篇】资料切片的内容作为参考。>\n"]
        
        for i, doc in enumerate(documents, 1):
            markdown_parts.append(f"#### 📄 切片 {i}")
            
            # 处理内容
            content = self._truncate_content(doc.page_content)
            markdown_parts.append(f"##### 切片内容")
            markdown_parts.append(f"```\n{content}\n```")
            
            # 处理元数据
            if doc.metadata:
                markdown_parts.append(f"##### 切片元数据")
                for key, value in doc.metadata.items():
                    markdown_parts.append(f"- **{key}**: `{value}`")
            
            markdown_parts.append("---\n")
        
        return "\n".join(markdown_parts)
    
    def to_json_string(self, documents: List[Any], pretty: bool = True) -> str:
        """将Document列表转换为JSON字符串"""
        doc_dicts = []
        for doc in documents:
            doc_dict = {
                "page_content": self._truncate_content(doc.page_content),
                "metadata": doc.metadata
            }
            doc_dicts.append(doc_dict)
        
        if pretty:
            return json.dumps(doc_dicts, indent=2, ensure_ascii=False)
        return json.dumps(doc_dicts, ensure_ascii=False)
    
    def to_summary_table(self, documents: List[Any]) -> str:
        """生成文档摘要表格"""
        if not documents:
            return "No documents to display"
        
        # 计算列宽
        max_content_width = min(50, self.max_content_length)
        
        # 表头
        header = f"{'#':<3} {'Content Preview':<{max_content_width}} {'Metadata Keys':<30}"
        separator = "-" * len(header)
        
        lines = [header, separator]
        
        for i, doc in enumerate(documents, 1):
            # 内容预览
            content_preview = doc.page_content.replace('\n', ' ').strip()
            if len(content_preview) > max_content_width - 3:
                content_preview = content_preview[:max_content_width-3] + "..."
            
            # 元数据键
            metadata_keys = ", ".join(doc.metadata.keys()) if doc.metadata else "None"
            if len(metadata_keys) > 27:
                metadata_keys = metadata_keys[:27] + "..."
            
            line = f"{i:<3} {content_preview:<{max_content_width}} {metadata_keys:<30}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _truncate_content(self, content: str) -> str:
        """截断内容到指定长度"""
        if len(content) <= self.max_content_length:
            return content
        return content[:self.max_content_length] + "\n[...省略]"


# Rich格式化器（需要安装rich库）
class RichDocumentDisplay:
    """使用Rich库的高级显示器"""
    
    def __init__(self):
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.syntax import Syntax
            from rich.columns import Columns
            from rich import box
            from rich.text import Text
            
            self.console = Console()
            self.Panel = Panel
            self.Table = Table
            self.Columns = Columns
            self.box = box
            self.Text = Text
            
        except ImportError:
            raise ImportError("Please install rich library: pip install rich")
    
    def display(self, documents: List[Any], style: str = "panel"):
        """显示文档，支持不同样式"""
        if style == "panel":
            self._display_as_panels(documents)
        elif style == "table":
            self._display_as_table(documents)
        elif style == "columns":
            self._display_as_columns(documents)
    
    def _display_as_panels(self, documents: List[Any]):
        """以面板形式显示"""
        for i, doc in enumerate(documents, 1):
            # 内容面板
            content = doc.page_content
            if len(content) > 400:
                content = content[:400] + "\n[...truncated]"
            
            content_panel = self.Panel(
                content,
                title=f"📄 Document {i}",
                subtitle=f"Length: {len(doc.page_content)} chars",
                border_style="blue"
            )
            
            # 元数据表格
            if doc.metadata:
                metadata_table = self.Table(box=self.box.MINIMAL)
                metadata_table.add_column("Key", style="cyan")
                metadata_table.add_column("Value", style="yellow")
                
                for key, value in doc.metadata.items():
                    metadata_table.add_row(str(key), str(value))
                
                metadata_panel = self.Panel(
                    metadata_table,
                    title="📊 Metadata",
                    border_style="green"
                )
                
                self.console.print(self.Columns([content_panel, metadata_panel]))
            else:
                self.console.print(content_panel)
            
            self.console.print()
    
    def _display_as_table(self, documents: List[Any]):
        """以表格形式显示"""
        table = self.Table(title=f"📚 Documents Collection ({len(documents)} items)")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Content Preview", style="white", width=50)
        table.add_column("Metadata", style="yellow", width=30)
        
        for i, doc in enumerate(documents, 1):
            content_preview = doc.page_content.replace('\n', ' ')[:50] + "..."
            metadata_str = str(doc.metadata) if doc.metadata else "None"
            if len(metadata_str) > 30:
                metadata_str = metadata_str[:27] + "..."
            
            table.add_row(str(i), content_preview, metadata_str)
        
        self.console.print(table)
    
    def _display_as_columns(self, documents: List[Any]):
        """以列形式显示"""
        panels = []
        for i, doc in enumerate(documents, 1):
            content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            metadata_text = ""
            if doc.metadata:
                metadata_text = "\n".join([f"{k}: {v}" for k, v in doc.metadata.items()])
            
            panel_content = f"{content}\n\n📊 Metadata:\n{metadata_text}" if metadata_text else content
            
            panel = self.Panel(
                panel_content,
                title=f"Doc {i}",
                width=40
            )
            panels.append(panel)
        
        self.console.print(self.Columns(panels, equal=True, expand=True))


# 使用示例函数
def demo_usage():
    """演示如何使用这些工具"""
    # 假设你有一个Document列表
    # documents = [your_document_list]
    
    # 方法1: 基础格式化器
    formatter = DocumentDisplayFormatter(max_content_length=200)
    
    # 生成Markdown
    # markdown_output = formatter.to_markdown(documents)
    # print(markdown_output)
    
    # 生成摘要表格
    # summary_table = formatter.to_summary_table(documents)
    # print(summary_table)
    
    # 生成JSON
    # json_output = formatter.to_json_string(documents)
    # print(json_output)
    
    # 方法2: Rich显示器（如果安装了rich）
    try:
        rich_display = RichDocumentDisplay()
        # rich_display.display(documents, style="panel")
        # rich_display.display(documents, style="table")
        # rich_display.display(documents, style="columns")
    except ImportError:
        print("Rich library not installed. Using basic formatter only.")

if __name__ == "__main__":
    demo_usage()