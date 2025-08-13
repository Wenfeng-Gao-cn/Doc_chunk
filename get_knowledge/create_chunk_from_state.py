"""
create_chunk_from_state.py - 将state中的chunk_list数据嵌入到向量数据库中

功能：
1. 从setup.yaml读取配置信息
2. 从state中读取chunk_list中的知识切片数据
3. 使用embedding模型将数据向量化
4. 存储到向量数据库中
"""

import os
import sys
import yaml
import json
from typing import List, Dict, Any, Optional
from Utils.graph_state import ChunkList, GraphState, KnowledgeTree
from Utils.logger import setup_logger
# LangChain imports - 使用新的包
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_chroma import Chroma

# 支持第三方API的自定义嵌入类
import requests
import time


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


class VectorDBManager:
    """向量数据库管理器"""
    
    def __init__(self, config_path: str = "setup.yaml"):
        """
        初始化向量数据库管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.embeddings = self._init_embeddings()
        self.vectorstore: Optional[Chroma] = None
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"配置文件未找到: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise
    
    def _init_embeddings(self) -> Embeddings:
        """初始化embedding模型，支持OpenAI和第三方API"""
        embedding_config = self.config.get('embedding_model', {})
        model_name = embedding_config.get('model_name', 'text-embedding-ada-002')
        # 支持多种API key字段名
        api_key = (embedding_config.get('api_key') or 
                  embedding_config.get('openai_api_key') or '')
        # 支持多种API base字段名
        api_base = (embedding_config.get('api_base') or 
                   embedding_config.get('openai_api_base') or '')
        max_retries = int(embedding_config.get('max_retries', 3))
        
        # 判断是否使用OpenAI模型
        if self._is_openai_model(model_name):
            logger.info(f"使用OpenAI兼容的embedding模型: {model_name}")
            # 使用新的langchain-openai包
            embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key if api_key else None,
                base_url=api_base if api_base else None,
                max_retries=max_retries
            )
        else:
            logger.info(f"使用第三方embedding模型: {model_name}")
            embeddings = CustomEmbeddings(
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                max_retries=max_retries
            )
        
        logger.info(f"成功初始化embedding模型: {model_name}")
        return embeddings
    
    def _is_openai_model(self, model_name: str) -> bool:
        """判断是否为OpenAI模型"""
        openai_models = [
            'text-embedding-ada-002',
            'text-embedding-3-small',
            'text-embedding-3-large',
            'text-davinci-003',
            'text-curie-001',
            'text-babbage-001',
            'text-ada-001'
        ]
        return model_name in openai_models
    
    def _init_vectorstore(self, persist_directory: Optional[str] = None) -> Chroma:
        """初始化向量数据库"""
        # 从配置文件获取向量数据库配置
        vectordb_config = self.config.get('vectordb_config', {})
        if persist_directory is None:
            persist_directory = str(vectordb_config.get('persist_directory', './chroma_db'))
        collection_name = str(vectordb_config.get('collection_name', 'xty_qa_collection'))
        
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        logger.info(f"初始化向量数据库: {persist_directory}, 集合名: {collection_name}")
        return self.vectorstore
    
    def create_documents_from_chunks(self, state: GraphState) -> List[Document]:
        """将state中的chunk_list转换为LangChain Document对象
        
        每个chunk作为一个独立的Document对象
        
        Args:
            state: GraphState对象，包含chunk_list
            
        Returns:
            List[Document]: Document对象列表
        """
        documents = []
        
        # 获取chunk列表
        chunks = state.chunk_list.chunks
        
        # 获取最大处理记录数配置
        vectordb_config = self.config.get('vectordb_config', {})
        max_records_config = vectordb_config.get('max_records', 0)
        max_records = int(max_records_config) if max_records_config is not None else 0
        
        # 如果设置了max_records且大于0，则限制处理的记录数
        if max_records > 0 and len(chunks) > max_records:
            chunks = chunks[:max_records]
            logger.info(f"根据配置限制，只处理前 {max_records} 个切片")
        
        for index, chunk in enumerate(chunks):
            try:
                # 跳过空切片
                if not chunk.title or not chunk.content:
                    logger.warning(f"跳过第 {index+1} 个切片：标题或内容为空")
                    continue
                
                # 每个chunk作为一个完整的内容
                # 组合方式：将标题和内容结构化组合
                content = f"""标题: {chunk.title}
内容: {chunk.content}"""
                
                # 创建详细的元数据
                metadata = {
                    'chunk_title': str(chunk.title),
                    'chunk_content': str(chunk.content),
                    'chunk_index': int(index),
                    'source': '知识切片',
                    'source_file': str(state.source_file) if state.source_file else '未知文件',
                    'source_doc_length': int(len(state.source_doc)) if state.source_doc else 0,
                    'knowledge_tree_title': str(state.knowledge_trees.title) if state.knowledge_trees and state.knowledge_trees.title else '',
                    'chunk_type': 'knowledge_chunk',  # 标识这是知识切片
                    'content_length': int(len(content))
                }
                
                # 如果chunk有自定义metadata，合并进去
                if chunk.metadata:
                    metadata.update({k: str(v) for k, v in chunk.metadata.items()})
                
                # 创建Document对象
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                
                documents.append(doc)
                
            except Exception as e:
                logger.error(f"处理第 {index+1} 个切片时出错: {e}")
                continue
        
        logger.info(f"成功创建了 {len(documents)} 个Document对象，每个对应一个知识切片")
        return documents
    
    def embed_documents(self, documents: List[Document], batch_size: Optional[int] = None) -> None:
        """将文档嵌入到向量数据库
        
        每个Document对象（对应一个知识切片）作为一个独立的chunk进行嵌入
        
        Args:
            documents: Document对象列表
            batch_size: 批处理大小，如果为None则从配置文件读取
        """
        try:
            if not self.vectorstore:
                logger.info("初始化向量数据库...")
                self._init_vectorstore()
            
            if self.vectorstore is None:
                logger.error("向量数据库初始化失败")
                raise ValueError("向量数据库初始化失败")
                
            # 获取配置中的批处理大小
            vectordb_config = self.config.get('vectordb_config', {})
            batch_size_config = vectordb_config.get('batch_size', 50)
            batch_size = int(batch_size_config) if batch_size is None else int(batch_size)
            
            logger.info(f"开始嵌入文档到向量数据库")
            logger.info(f"总文档数: {len(documents)}")
            logger.info(f"批处理大小: {batch_size}")
            logger.info(f"向量数据库路径: {getattr(self.vectorstore, '_persist_directory', '未知路径')}")
            logger.debug(f"第一个文档内容: {documents[0].page_content[:200]}...")
            logger.debug(f"第一个文档元数据: {documents[0].metadata}")
            
            # 分批处理文档
            total_docs = len(documents)
            success_count = 0
            error_count = 0
            
            for i in range(0, total_docs, batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    logger.info(f"处理批次 {batch_num}: 文档 {i+1}-{min(i + batch_size, total_docs)}")
                    logger.debug(f"批次 {batch_num} 第一个文档ID: {batch_docs[0].metadata.get('chunk_index', 'unknown')}")
                    
                    # 添加文档到向量数据库
                    self.vectorstore.add_documents(batch_docs)
                    success_count += len(batch_docs)
                    logger.info(f"批次 {batch_num} 处理成功，嵌入 {len(batch_docs)} 个切片")
                    
                except Exception as e:
                    error_count += len(batch_docs)
                    logger.error(f"批次 {batch_num} 处理失败: {str(e)}", exc_info=True)
                    logger.error(f"失败批次的第一个文档内容: {batch_docs[0].page_content[:200]}...")
                    
                    # 尝试单个文档处理（错误恢复）
                    logger.info(f"尝试单个文档处理批次 {batch_num}")
                    for j, doc in enumerate(batch_docs):
                        try:
                            logger.debug(f"处理单个文档 {j+1}/{len(batch_docs)}")
                            self.vectorstore.add_documents([doc])
                            success_count += 1
                            error_count -= 1
                        except Exception as single_error:
                            logger.error(f"单个文档处理失败 (切片索引 {doc.metadata.get('chunk_index', 'unknown')}): {str(single_error)}", exc_info=True)
                            logger.error(f"失败文档内容: {doc.page_content[:200]}...")
                
            # 新版本的Chroma不需要手动persist，数据会自动持久化
            logger.info("向量数据库数据已自动持久化")
            
            # 输出处理结果统计
            logger.info(f"=== 嵌入处理完成 ===")
            logger.info(f"成功嵌入: {success_count} 个切片")
            logger.info(f"处理失败: {error_count} 个切片")
            logger.info(f"总计处理: {total_docs} 个切片")
            
            if error_count > 0:
                raise ValueError(f"部分文档嵌入失败，共 {error_count} 个失败")
                
        except Exception as e:
            logger.error(f"文档嵌入过程中发生严重错误: {str(e)}", exc_info=True)
            logger.error(f"当前向量数据库状态: {self.vectorstore.__dict__ if self.vectorstore else '未初始化'}")
            raise
    
    def process_state_to_vectordb(self, state: GraphState) -> None:
        """完整的处理流程：从state到向量数据库
        
        Args:
            state: GraphState对象
            
        Returns:
            None
        """
        try:
            # 1. 验证state中的数据
            logger.info("=== 开始处理state中的知识切片到向量数据库 ===")
            logger.info(f"完整state结构: {state.model_dump_json(indent=2)}")
            
            # 验证必需字段
            if not hasattr(state, 'chunk_list') or not state.chunk_list:
                logger.error("state中缺少chunk_list字段或为空")
                raise ValueError("Invalid state: missing chunk_list")
                
            if not state.chunk_list.chunks:
                logger.warning("state中没有知识切片数据")
                return
                
            # 记录详细状态信息
            logger.info(f"源文件: {state.source_file}")
            logger.info(f"知识树标题: {state.knowledge_trees.title if state.knowledge_trees else '无'}")
            logger.info(f"切片数量: {len(state.chunk_list.chunks)}")
            logger.info(f"前3个切片标题: {[chunk.title for chunk in state.chunk_list.chunks[:3]]}")
            
            # 2. 创建Document对象
            logger.info("开始创建Document对象...")
            documents = self.create_documents_from_chunks(state)
            
            if not documents:
                logger.warning("没有有效的文档可以处理")
                return
                
            logger.info(f"成功创建 {len(documents)} 个Document对象")
            logger.debug(f"第一个Document内容: {documents[0].page_content[:200]}...")
            logger.debug(f"第一个Document元数据: {documents[0].metadata}")
            
            # 3. 嵌入到向量数据库
            logger.info("开始嵌入到向量数据库...")
            self.embed_documents(documents)
            
            logger.info("=== state知识切片处理完成 ===")
            
        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
            logger.error(f"错误发生时state内容: {state.model_dump_json() if hasattr(state, 'model_dump_json') else str(state)}")
            raise ValueError(f"处理state到向量数据库失败: {str(e)}") from e
    
    def test_embedding_connection(self) -> bool:
        """测试嵌入模型连接是否正常"""
        try:
            logger.info("正在测试embedding模型连接...")
            test_text = "这是一个测试文本"
            
            # 尝试嵌入测试文本
            embedding = self.embeddings.embed_query(test_text)
            
            if embedding and len(embedding) > 0:
                logger.info(f"embedding模型连接成功，向量维度: {len(embedding)}")
                return True
            else:
                logger.error("embedding模型返回空向量")
                return False
                
        except Exception as e:
            logger.error(f"embedding模型连接测试失败: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """搜索相似文档（用于测试）"""
        if not self.vectorstore:
            self._init_vectorstore()
            
        if self.vectorstore is None:
            logger.error("向量数据库初始化失败")
            return []
            
        # 将query与所有metadata字段组合进行搜索
        metadata_fields = [
            "chunk_title", "chunk_content", "source_file", 
            "knowledge_tree_title", "chunk_type"
        ]
        
        # 为每个metadata字段创建增强查询
        enhanced_queries = [query]
        for field in metadata_fields:
            enhanced_queries.append(f"{field}:{query}")
            
        # 合并所有增强查询
        combined_query = " ".join(enhanced_queries)
        
        return self.vectorstore.similarity_search(combined_query, k=k)


def main():
    """主函数"""
    try:
        # 检查配置文件是否存在
        if not os.path.exists("setup.yaml"):
            logger.error("setup.yaml配置文件不存在")
            return
        
        # 创建向量数据库管理器
        vector_manager = VectorDBManager("setup.yaml")
        
        # 测试embedding模型连接
        if not vector_manager.test_embedding_connection():
            logger.error("embedding模型连接失败，请检查配置")
            return
        
        # 从json文件读取state数据
        json_path = "sample_doc/test_gen_k_chunk_output.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # 创建state对象
        sample_state = GraphState(
            knowledge_trees=KnowledgeTree(**state_data["knowledge_trees"]),
            source_doc=state_data["source_doc"] or "",
            source_file=state_data["source_file"] or "",
            chunk_list=ChunkList(**state_data["chunk_list"])
        )
        
        # 执行完整的处理流程
        vector_manager.process_state_to_vectordb(sample_state)
        
        # 可选：测试搜索功能
        print("\n=== 测试搜索功能 ===")
        # test_query = "资费公示"
        test_query = input("请输入需要搜索的内容：")
        results = vector_manager.search_similar(test_query, k=3)
        
        print(f"搜索查询: {test_query}")
        print(f"返回 {len(results)} 个最相似的切片:")
        
        for i, doc in enumerate(results, 1):
            print(f"\n--- 结果 {i} ---")
            print(f"切片标题: {doc.metadata.get('chunk_title')}")
            chunk_content = doc.metadata.get('chunk_content', '')
            print(f"切片内容: {chunk_content[:100]}{'...' if len(chunk_content) > 100 else ''}")
            print(f"切片索引: {doc.metadata.get('chunk_index')}")
            print(f"内容长度: {doc.metadata.get('content_length')} 字符")
            print(f"源文件: {doc.metadata.get('source_file')}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
