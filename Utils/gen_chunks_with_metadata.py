from pydantic import BaseModel, Field
from Utils.load_setup import load_setup
from Utils.logger import setup_logger
from Utils.llm import get_llm_from_list
from openai import BadRequestError
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts  import PromptTemplate
from rich import print
import asyncio

logger = setup_logger(__name__)

class context(BaseModel):
    topic: str = Field(..., description="文档片段的主题")
    keywords: list['str']= Field(..., description="安装文档的关键词")
    entities: list['str'] = Field(..., description="文档涉及的重要实体")
    question : list['str'] = Field(..., description="安装文档常见问题")
    background : str = Field(..., description="文档背景")

class metadata(BaseModel):
    context: context
    source_file: str

class chunk(BaseModel):
    metadata: metadata
    chunk_content: str


async def gen_chunks_with_metadata(file_name:str,source_doc:str,chunk_list:list) -> list:
    chunklist_w_metadata = []
    print("-------开始为切片添加元数据-------")
    for chunk in chunk_list:
        chunk_w_metadata = await add_metadata_2_chunk(
            source_file = file_name,
            source_doc = source_doc,
            chunk_str = chunk
            )
        print(f"\n添加了元数据的切片结果：\n{chunk_w_metadata}\n")
        chunklist_w_metadata.append(chunk_w_metadata)
    return chunklist_w_metadata


async def add_metadata_2_chunk(source_file:str,source_doc:str,chunk_str:str) -> chunk:
    config=load_setup()
    prompt_file = config.get("graph_config",{}).get("gen_metadata_prompt")
    try: 
        with open(prompt_file,'r',encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {prompt_file}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt file: {e}")
        raise
    parser = JsonOutputParser(pydantic_object=context)
    format_instructions = parser.get_format_instructions()
    input_2_llm = PromptTemplate(
        input_variables = ["source_doc","chunk_str"],
        template = prompt_template,
        partial_variables = {"format_instructions": format_instructions}
    )
    
    last_error = None
    seq = 0
    while True:
        try:
            chain = input_2_llm | get_llm_from_list("gen_metadata_llm", seq) | parser
            result_dict = chain.invoke({
                "source_doc": source_doc,
                "chunk_str": chunk_str
            })
            break
        except IndexError:
            if last_error:
                logger.error(f"所有LLM配置尝试失败，最后一个错误: {str(last_error)}")
                print(f"所有LLM配置尝试失败，最后一个错误: {str(last_error)}")
                raise last_error
            raise
        except Exception as e:
            last_error = e
            logger.warning(f"LLM调用失败(seq={seq}): {str(e)}，尝试下一个配置...")
            print(f"LLM调用失败(seq={seq}): {str(e)}，尝试下一个配置...")
            seq += 1

    result = context(**result_dict)

    chunk_metadata = metadata(
        context=result,
        source_file=source_file
    ) 

    result_chunk = chunk(
        metadata = chunk_metadata,
        chunk_content = chunk_str
    )

    return result_chunk

if __name__ == "__main__":

    file_name = "Robot4.0安装说明文档V1.8.docx"
    from Utils.readfile_2_str import read_file_to_string
    source_doc = read_file_to_string("sample_doc\云趣运维文档1754623245145\语音机器人安装手册V1.7.pdf")
    chunk_list=["""
server {
        listen 7773;  # 监听外部访问的端口，可以改为其他端口
        server_name 172.16.80.241;  # 替换为你的服务器域名或IP地址

        location / {
            proxy_pass http://127.0.0.1:8293;  # classifier_emotion配置的server_port
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
"""]
    asyncio.run(gen_chunks_with_metadata(file_name,source_doc,chunk_list))
