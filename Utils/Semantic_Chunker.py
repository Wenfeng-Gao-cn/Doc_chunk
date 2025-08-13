from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from Utils.load_setup import load_setup
from Utils.embeddings import get_embeddings
from Utils.load_setup import load_setup


def semantic_chunker (inputstr:str)->list:
    embeddings = get_embeddings()
    setup = load_setup()
    splitter_config = setup.get("Splitter_config",{})
    recursive_config = splitter_config.get("Recursive_config",{})
    sentence_transformers_config = splitter_config.get("SentenceTransformers_config",{})

    pre_splitter = RecursiveCharacterTextSplitter(**recursive_config)  # 预分割
    
    # 2. 语义分块 (动态边界检测)
    semantic_splitter = SemanticChunker(
        embeddings, 
        **sentence_transformers_config
        )
    
    # 3. 后处理修复 (连接被截断的句子)
    final_chunks = []
    chunks = pre_splitter.split_text(inputstr)
    for chunk in chunks:
        sub_chunks = semantic_splitter.split_text(chunk)
        # 合并首尾不完整句子
        if final_chunks and final_chunks[-1].endswith(("，", "；")): 
            final_chunks[-1] += sub_chunks[0]
            sub_chunks = sub_chunks[1:]
        final_chunks.extend(sub_chunks)
    return final_chunks

if __name__ == "__main__" :
    testfile="sample_doc/云趣运维文档1754623245145/Robot4.0安装说明文档V1.8.docx"
    from Utils.readfile_2_str import read_file_to_string
    from rich import print
    test_content = read_file_to_string(testfile)
    print(f"内容长度: {len(test_content)} 字符")
    print(f"内容预览: {test_content[:100]}...")
    print("--------开始进行语义分割--------")
    chunks = semantic_chunker(test_content)
    for chunk in chunks:
        print(f"\n--------\n{chunk}\n--------\n")
    
