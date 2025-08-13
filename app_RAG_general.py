"""
app_RAG_general.py

功能：
1. 从setup.yaml读取配置信息
2. 通过嵌入模型进行初步切片
3. 对初步切片的结果加入metadata后变成完整的切片信息
3. 使用embedding模型将数据向量化
4. 存储到向量数据库中
"""

import os
import sys
import yaml
import asyncio
import json
from typing import List, Dict, Any, Optional
from Utils.graph_state import ChunkList, GraphState, KnowledgeTree
from Utils.logger import setup_logger
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from Utils.connect_embeddings import CustomEmbeddings
from Utils.readfile_2_str import read_file_to_string
from Utils.Semantic_Chunker import semantic_chunker
from Utils.gen_chunks_with_metadata import gen_chunks_with_metadata

logger = setup_logger(__name__)

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
    
    def process_state_to_vectordb(self, chunks: list) -> None:
        """处理知识切片列表到向量数据库
        
        Args:
            chunks: 知识切片列表，每个元素包含metadata和chunk_content
            
        Returns:
            None
        """
        try:
            logger.info("=== 开始处理知识切片到向量数据库 ===")
            
            if not chunks:
                logger.warning("没有知识切片数据")
                return
                
            # 记录基本信息
            logger.info(f"切片数量: {len(chunks)}")
            source_file = chunks[0].metadata.source_file if hasattr(chunks[0], 'metadata') else '未知'
            logger.info(f"第一个切片的源文件: {source_file}")
            
            # 初始化向量数据库
            if not self.vectorstore:
                self._init_vectorstore()
                
            # 删除同源文件的旧数据
            if self.vectorstore and source_file != '未知':
                logger.info(f"正在删除向量数据库中源文件为 {source_file} 的旧数据...")
                try:
                    self.vectorstore.delete(where={"source_file": source_file})
                    logger.info(f"成功删除源文件 {source_file} 的旧数据")
                except Exception as e:
                    logger.error(f"删除旧数据失败: {e}")
                    raise
                    
            # 创建Document对象
            logger.info("开始创建Document对象...")
            documents = []
            
            for index, chunk in enumerate(chunks):
                try:
                    if not hasattr(chunk, 'chunk_content') or not chunk.chunk_content:
                        logger.warning(f"跳过第 {index+1} 个切片：内容为空")
                        continue
                        
                    # 创建元数据
                    metadata = {
                        'chunk_content': str(chunk.chunk_content),
                        'chunk_index': int(index),
                        'source_file': str(chunk.metadata.source_file) if hasattr(chunk.metadata, 'source_file') else '未知文件',
                        'topic': str(chunk.metadata.context.topic) if hasattr(chunk.metadata, 'context') else '',
                        'keywords': str(chunk.metadata.context.keywords) if hasattr(chunk.metadata, 'context') else '',
                        'entities': str(chunk.metadata.context.entities) if hasattr(chunk.metadata, 'context') else '',
                        'chunk_type': 'knowledge_chunk',
                        'content_length': int(len(chunk.chunk_content))
                    }
                    
                    # 创建Document对象
                    doc = Document(
                        page_content=str(chunk.chunk_content),
                        metadata=metadata
                    )
                    
                    documents.append(doc)
                    
                except Exception as e:
                    logger.error(f"处理第 {index+1} 个切片时出错: {e}")
                    continue
                    
            if not documents:
                logger.warning("没有有效的文档可以处理")
                return
                
            logger.info(f"成功创建 {len(documents)} 个Document对象")
            logger.debug(f"第一个Document内容: {documents[0].page_content[:200]}...")
            logger.debug(f"第一个Document元数据: {documents[0].metadata}")
            
            # 嵌入到向量数据库
            logger.info("开始嵌入到向量数据库...")
            self.embed_documents(documents)
            
            logger.info("=== 知识切片处理完成 ===")
            
        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
            raise ValueError(f"处理知识切片到向量数据库失败: {str(e)}") from e
    
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


async def gen_single_file_chunks(file:str):
    try:
        doc_str = read_file_to_string(file)
        chunks = semantic_chunker(doc_str)
        # 提取纯文件名（不含路径）
        file_name = os.path.basename(file)
        chunks_w_metadata = await gen_chunks_with_metadata(file_name,doc_str,chunks)
        vector_manager = VectorDBManager("setup.yaml")
        # 测试embedding模型连接
        if not vector_manager.test_embedding_connection():
            logger.error("embedding模型连接失败，请检查配置")
            print("embedding模型连接失败，请检查配置")
            return
        vector_manager.process_state_to_vectordb(chunks_w_metadata)
        print(f"成功把文件{file}切片,并嵌入向量数据库，持久化完成")
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        print(f"程序执行失败: {e}")
        sys.exit(1)

async def gen_chunks_from_folder(folder_name:str="sample_doc"):
    """处理文件夹中的所有支持的文件
    
    Args:
        folder_name: 文件夹路径
    """
    supported_extensions = ['.txt', '.md', '.docx', '.doc', '.xlsx', '.xls', '.pdf']
    
    try:
        # 检查文件夹是否存在
        if not os.path.isdir(folder_name):
            logger.error(f"文件夹不存在: {folder_name}")
            print(f"文件夹不存在: {folder_name}")
            return

        # 遍历文件夹中的所有文件
        for root, _, files in os.walk(folder_name):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # 只处理支持的文件格式
                if file_ext in supported_extensions:
                    try:
                        print(f"正在处理文件: {file_path}")
                        await gen_single_file_chunks(file_path)
                    except Exception as e:
                        logger.error(f"处理文件 {file_path} 失败: {e}")
                        print(f"处理文件 {file_path} 失败: {e}")
                        continue
                    
        print(f"完成文件夹 {folder_name} 中所有支持文件的处理")
    except Exception as e:
        logger.error(f"处理文件夹 {folder_name} 时发生错误: {e}")
        print(f"处理文件夹 {folder_name} 时发生错误: {e}")
        raise

async def main():
    file_name="sample_doc\云趣运维文档1754623245145\语音机器人安装手册V1.7.pdf"
    await gen_single_file_chunks(file_name)

if __name__ == "__main__":
    asyncio.run(main())
