from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from Utils.load_setup import load_setup
from typing import List, Sequence, Any
import openai
import json
from Utils.logger import setup_logger
from pydantic import Field, PrivateAttr

logger = setup_logger(__name__)

class CustomReranker(BaseDocumentCompressor):
    """自定义重排器，使用API调用重排模型"""
    
    model_name: str = Field(description="重排模型名称")
    api_key: str = Field(description="API密钥")
    api_base: str = Field(description="API基础URL")
    max_retries: int = Field(default=3, description="最大重试次数")
    request_timeout: int = Field(default=60, description="请求超时时间")
    
    # 使用私有属性存储客户端
    _client: Any = PrivateAttr()
    
    def model_post_init(self, __context) -> None:
        """Pydantic v2 的初始化后处理方法"""
        super().model_post_init(__context)
        
        # 初始化OpenAI客户端
        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    @property
    def client(self):
        """获取OpenAI客户端"""
        return self._client
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        """对文档进行重排序"""
        if not documents:
            return documents
        
        try:
            # 准备重排请求的数据
            passages = [doc.page_content for doc in documents]
            
            # 调用重排模型API
            scores = self._rerank(query, passages)
            
            # 将文档和分数配对，并按分数降序排序
            doc_score_pairs = list(zip(documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # 返回重排后的文档，并在metadata中添加重排分数
            reranked_docs = []
            for doc, score in doc_score_pairs:
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, 'rerank_score': score}
                )
                reranked_docs.append(new_doc)
            
            logger.info(f"重排完成，返回 {len(reranked_docs)} 个文档")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"重排过程中出现错误: {e}")
            # 如果重排失败，返回原始文档
            return documents
    
    def _rerank(self, query: str, passages: List[str]) -> List[float]:
        """调用重排模型API获取分数"""
        try:
            # 构造重排请求
            messages = []
            
            # 系统提示词
            system_prompt = """你是一个文档相关性评分专家。给定一个查询和多个文档片段，请为每个文档片段与查询的相关性打分。
分数范围：0-1，其中1表示完全相关，0表示完全不相关。
请只返回分数列表，格式为JSON数组，例如：[0.95, 0.82, 0.71]"""
            
            messages.append({"role": "system", "content": system_prompt})
            
            # 用户提示词
            user_prompt = f"""查询：{query}\n\n文档片段：\n"""
            for i, passage in enumerate(passages):
                user_prompt += f"{i+1}. {passage}\n\n"
            
            user_prompt += "请为上述每个文档片段与查询的相关性打分，返回JSON格式的分数数组："
            messages.append({"role": "user", "content": user_prompt})
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=100,
                timeout=self.request_timeout
            )
            
            # 解析响应
            response_text = response.choices[0].message.content
            if response_text is None:
                logger.warning("重排模型返回空响应")
                return [1.0 / (i + 1) for i in range(len(passages))]
            response_text = response_text.strip()
            
            # 尝试解析JSON
            try:
                scores = json.loads(response_text)
                if len(scores) != len(passages):
                    logger.warning(f"返回的分数数量({len(scores)})与文档数量({len(passages)})不匹配")
                    # 如果数量不匹配，使用默认分数
                    scores = [1.0 / (i + 1) for i in range(len(passages))]
            except json.JSONDecodeError:
                logger.warning(f"无法解析重排模型返回的JSON: {response_text}")
                # 解析失败时使用默认分数（按原始顺序递减）
                scores = [1.0 / (i + 1) for i in range(len(passages))]
            
            return scores
            
        except Exception as e:
            logger.error(f"调用重排模型API时出错: {e}")
            # API调用失败时，返回默认分数
            return [1.0 / (i + 1) for i in range(len(passages))]
