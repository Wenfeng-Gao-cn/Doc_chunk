from Utils.logger import setup_logger
from langchain.embeddings.base import Embeddings
import requests
import time
from typing import List

logger = setup_logger(__name__)
class CustomEmbeddings(Embeddings):
    """自定义嵌入类，支持第三方API（如SiliconFlow等）"""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base: str,
        max_retries: int = 3,
        request_timeout: int = 60
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        
        # 设置请求头
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """发送嵌入请求到API"""
        url = f"{self.api_base}/v1/embeddings"
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = []
                    
                    # 提取嵌入向量
                    for item in result.get('data', []):
                        embeddings.append(item.get('embedding', []))
                    
                    return embeddings
                else:
                    logger.warning(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
        
        raise Exception(f"经过 {self.max_retries} 次重试后，嵌入请求仍然失败")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return self._make_request(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        result = self._make_request([text])
        return result[0] if result else []
