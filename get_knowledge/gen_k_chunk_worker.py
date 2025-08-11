import yaml
import asyncio
from Utils.graph_state import GraphState, KnowledgeTree, KnowledgeNode, KnowledgeChunk, ChunkList
from pydantic import BaseModel
from pydantic.fields import Field
from Utils.logger import setup_logger
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts  import PromptTemplate
from Utils.llm import get_llm
from rich import print

logger = setup_logger(__name__)

class RecorrectRequest(BaseModel):
    content: str = Field(..., description="当前知识点的源文档的详细内容")


async def Recorrect_knowledge(source_doc: str, knowledge_chunk: KnowledgeChunk) -> KnowledgeChunk:
    # 这里是修正知识树的逻辑
    setup_file = "setup.yaml"
    try:
        with open(setup_file, encoding='utf-8') as f:
            config = yaml.safe_load(f)
            recorrect_prompt_file = config["graph_config"]["recorrect_k_prompt"]
            try:
                with open(recorrect_prompt_file, encoding='utf-8') as prompt_file:
                    recorrect_prompt = prompt_file.read()
            except FileNotFoundError:
                logger.error(f"Prompt file not found: {recorrect_prompt_file}")
                raise
            except Exception as e:
                logger.error(f"Error reading prompt file: {e}")
                raise
    except yaml.YAMLError as ye:
        logger.error(f"加载{setup_file}配置失败: {str(ye)}", exc_info=True)
        raise
    except KeyError as ke:
        logger.error(f"缺少配置键: {str(ke)} - 文件: {setup_file}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"加载{setup_file}配置失败: {str(e)}", exc_info=True)
        raise

    parser = JsonOutputParser(pydantic_object=RecorrectRequest)
    format_instructions = parser.get_format_instructions()
    input_2_llm = PromptTemplate(
        input_variables=["source_doc", "knowledge_chunk"],
        template=recorrect_prompt,
        partial_variables={"format_instructions": format_instructions}
    )

    chain = input_2_llm | get_llm("recorrect_k_llm") | parser
    result = chain.invoke({
        "source_doc": source_doc,
        "knowledge_chunk": knowledge_chunk
    })
    knowledge_chunk.content = result["content"]

    return knowledge_chunk

def init_chunk_list(k_tree: KnowledgeTree, source_file: str) -> ChunkList:
    """初始化切片列表，只对没有子节点的叶子节点生成切片"""
    chunks_list = []
    all_nodes = k_tree.get_all_nodes()
    
    for node_info in all_nodes:
        node = node_info["node"]
        path = node_info["path"]
        
        # 只有没有子节点的叶子节点才生成切片
        if not node.children:
            chunk = KnowledgeChunk(
                title=node.title,
                content=node.content,
                metadata={
                    "parent_path": path[:-1],  # 排除当前节点自身的标题
                    "full_path": path,
                    "source_file": source_file
                }
            )
            chunks_list.append(chunk)
        
    return ChunkList(chunks=chunks_list)

async def gen_knowledge_chunk(state: GraphState) -> ChunkList:
    """为切片列表中的每个切片的content更改为对应的原文内容"""
    chunk_list = init_chunk_list(state.knowledge_trees, state.source_file)
    print("开始生成的切片列表的内容修正:")
    for chunk in chunk_list.chunks:
        
        chunk.content = (await Recorrect_knowledge(state.source_doc, chunk)).content
        print(f"\n修正切片: {chunk.title}\n 修正后内容: {chunk.content}\n")
        print("========================================")
    return chunk_list

if __name__ == "__main__":
    source_doc_file = "sample_doc/11、《中华人民共和国电信条例》.txt"
    test_knowledge_trees_file = "sample_doc/test_eva_k_output.json"
    try:
        with open(source_doc_file, 'r', encoding='utf-8') as f:
            source_doc = f.read()
        with open(test_knowledge_trees_file, 'r', encoding='utf-8') as f:
            knowledge_trees = KnowledgeTree.model_validate(yaml.safe_load(f)["knowledge_trees"])
        state = GraphState(
            source_doc=source_doc,
            source_file=source_doc_file,
            knowledge_trees=knowledge_trees
        )
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"加载文件失败: {str(e)}", exc_info=True)
        raise
    async def main():
        result = await gen_knowledge_chunk(state)
        return GraphState(
            chunk_list=result
        )
    result = asyncio.run(main())
    print("生成的知识切片:")
    print(result.model_dump_json())
