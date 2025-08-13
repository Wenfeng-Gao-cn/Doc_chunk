from get_knowledge.create_chunk_from_state import VectorDBManager
from Utils.logger import setup_logger
import sys

logger = setup_logger(__name__)

try:
    vector_manager = VectorDBManager("setup.yaml")
    if not vector_manager.test_embedding_connection():
        logger.error("embedding模型连接失败，请检查配置")
        pass

    print("\n=== 测试搜索功能 ===")
    # test_query = "资费公示"
    test_query = input("请输入需要搜索的内容：")
    results = vector_manager.search_similar(test_query, k=3)

    print(f"搜索查询: {test_query}")
    print(f"返回 {len(results)} 个最相似的切片:")

    for i, doc in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print("所有元数据信息:")
        for key, value in doc.metadata.items():
            print(f"{key}: {value}")
        print("\n切片内容预览:")
        chunk_content = doc.metadata.get('chunk_content', '')
        print(chunk_content)

except Exception as e:
    logger.error(f"程序执行失败: {e}")
    sys.exit(1)
