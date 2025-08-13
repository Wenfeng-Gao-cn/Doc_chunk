"""clear_vector_db.py - 清除向量数据库集合内容"""

import os
import yaml
from typing import Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from Utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorDBCleaner:
    """向量数据库清理工具"""
    
    def __init__(self, config_path: str = "setup.yaml"):
        """
        初始化清理工具
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.embeddings = self._init_embeddings()
        self.vectorstore: Optional[Chroma] = None
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
            
    def _init_embeddings(self):
        """初始化embedding模型"""
        embedding_config = self.config.get('embedding_model', {})
        model_name = embedding_config.get('model_name', 'text-embedding-ada-002')
        api_key = embedding_config.get('openai_api_key', '')
        api_base = embedding_config.get('openai_api_base', '')
        
        return OpenAIEmbeddings(
            model=model_name,
            api_key=api_key if api_key else None,
            base_url=api_base if api_base else None
        )
        
    def _init_vectorstore(self) -> Chroma:
        """初始化向量数据库连接"""
        vectordb_config = self.config.get('vectordb_config', {})
        persist_directory = vectordb_config.get('persist_directory', './chroma_db')
        collection_name = vectordb_config.get('collection_name', 'xty_qa_collection')
        
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        return self.vectorstore
        
    def clear_collection(self) -> bool:
        """清除当前集合的所有内容"""
        try:
            if not self.vectorstore:
                self._init_vectorstore()
                
            if self.vectorstore is None:
                logger.error("向量数据库初始化失败")
                return False
                
            # 获取当前集合名称和持久化目录
            collection_name = self.vectorstore._collection.name
            persist_dir = self.config.get('vectordb_config', {}).get('persist_directory', './chroma_db')
            
            # 清除前记录数
            count_before = self.vectorstore._collection.count()
            logger.info(f"清除前集合记录数: {count_before}")
            
            # 删除集合
            self.vectorstore.delete_collection()
            logger.info(f"已成功清除集合: {collection_name}")
            
            # 重新创建空集合
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_dir,
                collection_name=collection_name
            )
            
            # 清除后记录数
            count_after = self.vectorstore._collection.count()
            logger.info(f"清除后集合记录数: {count_after}")
            
            print(f"\n清除结果:")
            print(f"清除前记录数: {count_before}")
            print(f"清除后记录数: {count_after}")
            
            return True
            
        except Exception as e:
            logger.error(f"清除集合失败: {e}")
            return False
            
def main():
    """主函数"""
    try:
        cleaner = VectorDBCleaner()
        if cleaner.clear_collection():
            print("向量数据库集合已成功清除并重置")
        else:
            print("清除集合失败")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"程序执行失败: {e}")

if __name__ == "__main__":
    main()
