import os
import asyncio
from pathlib import Path
from typing import List
import logging
from Utils.readfile_2_str import get_file_info

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 支持的扩展名
SUPPORTED_EXTENSIONS = ['.docx', '.doc', '.xlsx', '.xls', '.md', '.txt', '.pdf']

def get_supported_files(directory: str) -> List[str]:
    """获取目录中所有支持的文件
    
    Args:
        directory (str): 目录路径
        
    Returns:
        List[str]: 支持的文件路径列表
    """
    supported_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_info = get_file_info(file_path)
            
            if file_info['is_supported']:
                supported_files.append(file_path)
                logger.info(f"找到支持的文件: {file_path}")
            else:
                logger.warning(f"跳过不支持的文件: {file_path}")
    
    return supported_files

async def process_file(file_path: str):
    """处理单个文件
    
    Args:
        file_path (str): 文件路径
    """
    try:
        logger.info(f"开始处理文件: {file_path}")
        
        # 调用gen_chunk_graph.py处理文件
        from gen_chunk_graph import gen_chunk_graph
        result = await gen_chunk_graph(file_path)
        
        if result:
            logger.info(f"成功处理文件: {file_path}")
        else:
            logger.error(f"处理文件失败: {file_path}")
            
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")

async def process_files_in_directory(directory: str):
    """处理目录中的所有支持的文件
    
    Args:
        directory (str): 目录路径
    """
    # 获取支持的文件列表
    files = get_supported_files(directory)
    
    if not files:
        logger.warning(f"目录中没有支持的文件: {directory}")
        return
    
    logger.info(f"找到 {len(files)} 个支持的文件需要处理")
    
    # 顺序处理文件
    for file in files:
        await process_file(file)
    
    logger.info("所有文件处理完成")

if __name__ == "__main__":
    # 设置要处理的目录
    target_directory = "sample_doc/云趣运维文档1754623245145"
    
    try:
        logger.info(f"开始处理目录: {target_directory}")
        asyncio.run(process_files_in_directory(target_directory))
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.error(f"处理目录时发生错误: {str(e)}")
