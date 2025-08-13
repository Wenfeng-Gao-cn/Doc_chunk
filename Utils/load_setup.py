import yaml
from Utils.logger import setup_logger
from typing import Dict, Any

logger = setup_logger(__name__)

def load_setup(file_name="setup.yaml") -> Dict[str, Any]:
    try:
        with open(file_name,'r',encoding='utf-8') as f:
            setup = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {file_name}")
            return setup
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {file_name}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"配置文件格式错误: {e}")
        raise