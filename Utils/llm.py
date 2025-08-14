from langchain_openai import ChatOpenAI
import yaml
from rich import print
from Utils.logger import setup_logger

# 初始化logger
logger = setup_logger(__name__)

def get_llm(type:str="llm",json_ouput=False) -> ChatOpenAI:
    file= "setup.yaml"
    try:
        with open(file, encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if type != "llm":
                type= data["graph_config"][type]
            llm_params = data[type]
            if json_ouput:
                llm = ChatOpenAI(**llm_params, model_kwargs={"response_format": {"type": "json_object"}})
            else:
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

def get_llm_from_list(type: str, seq:int=0,json_ouput=False):
    file= "setup.yaml"
    try:
        with open(file, encoding='utf-8') as f:
            data = yaml.safe_load(f)
            type_value = data["graph_config"][type]
            # 处理type_value可能是列表或字符串的情况
            if isinstance(type_value, list):
                if seq >= len(type_value):
                    raise IndexError(f"seq参数{seq}超出范围(0-{len(type_value)-1})")
                llm_name = type_value[seq]
            else:
                if seq != 0:
                    raise IndexError(f"seq参数{seq}无效，非列表配置只支持seq=0")
                llm_name = type_value
            llm_params = data[llm_name]
            if json_ouput:
                llm = ChatOpenAI(**llm_params, model_kwargs={"response_format": {"type": "json_object"}})
            else:
                llm = ChatOpenAI(**llm_params)
            return llm
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
    try:
        llm=get_llm_from_list("gen_metadata_llm",1)
        print(f"llm的值：{llm}")

        print(f"llm.model_name的值：{llm.model_name}")

    except Exception as e:
        print(e)
