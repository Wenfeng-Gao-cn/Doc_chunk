from langchain_openai import ChatOpenAI
import yaml
from rich import print
from Utils.logger import setup_logger

# 初始化logger
logger = setup_logger(__name__)

def get_llm(type:str="llm") -> ChatOpenAI:
    file= "setup.yaml"
    try:
        with open(file, encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if type != "llm":
                type= data["graph_config"][type]
            llm_params = data[type]    
            llm = ChatOpenAI(**llm_params)
            return(llm)  
    except yaml.YAMLError as ye:
        logger.error(f"YAML解析失败: {str(ye)} - 文件: {file}", exc_info=True)
        raise
    except KeyError as ke:
        logger.error(f"配置键缺失: {str(ke)} - 文件: {file}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"加载LLM配置失败: {str(e)} - 文件: {file}", exc_info=True)
        raise

if __name__=="__main__":
    llm=get_llm("get_k_llm")
    print(f"get_llm的返回类型：{type(llm)}")
    print(f"get_llm的值：{llm}")
