import logging
import os
import yaml
from typing import Optional

def get_debug_config() -> bool:
    """从setup.yaml读取debug_logger配置"""
    try:
        with open('setup.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('debug_logger', False)
    except FileNotFoundError:
        logging.error("setup.yaml配置文件未找到")
        return False
    except yaml.YAMLError as e:
        logging.error(f"setup.yaml配置文件解析错误: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"读取debug_logger配置时发生未知错误: {str(e)}")
        return False

def setup_logger(name: str, debug_enabled: Optional[bool] = None) -> logging.Logger:
    """设置并返回配置好的logger实例
    
    Args:
        name: logger名称，通常使用__name__
        debug_enabled: 是否启用debug日志，None表示自动从配置读取
        
    Returns:
        配置好的Logger实例
    """
    # 确保logs目录存在
    os.makedirs('logs', exist_ok=True)
    
    # 配置ERROR级别日志
    logging.basicConfig(
        filename='logs/error.log',
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    
    # 如果未指定debug_enabled，则从配置读取
    if debug_enabled is None:
        debug_enabled = get_debug_config()
    
    # 添加INFO级别日志处理器（如果启用）
    if debug_enabled:
        info_handler = logging.FileHandler('logs/debug.log')
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(info_handler)
    
    return logger


if __name__ == "__main__":
    print("=== 测试logger模块 ===")
    
    # 测试1: 正常读取配置
    print("\n测试1 - 正常读取配置:")
    debug_enabled = get_debug_config()
    print(f"当前debug_logger配置值: {debug_enabled}")
    
    # 测试2: 测试setup_logger
    print("\n测试2 - 创建logger实例:")
    test_logger = setup_logger("test_logger")
    test_logger.error("这是一条ERROR级别日志")
    test_logger.info("这是一条INFO级别日志")  # 只有debug_enabled=True时才会记录
    
    # 测试3: 模拟配置文件不存在
    print("\n测试3 - 模拟配置文件不存在:")
    original_path = 'setup.yaml'
    temp_path = 'setup.yaml.bak'
    import os
    os.rename(original_path, temp_path)  # 临时重命名配置文件
    
    try:
        debug_enabled = get_debug_config()  # 应该会记录错误并返回False
        print(f"配置文件不存在时返回值: {debug_enabled}")
    finally:
        os.rename(temp_path, original_path)  # 恢复配置文件
    
    print("\n测试完成，请检查logs/error.log和logs/debug.log确认日志记录情况")
