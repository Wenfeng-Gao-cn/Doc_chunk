from get_knowledge.create_chunk_from_state import CustomEmbeddings
from Utils.load_setup import load_setup

def get_embeddings () ->CustomEmbeddings:
    config=load_setup()
    embedding_config = config.get("embedding_model",{})
    modle_name = embedding_config.get("model_name")
    api_base = embedding_config.get("openai_api_base")
    api_key = embedding_config.get("openai_api_key")
    embeddings = CustomEmbeddings(
        model_name = modle_name,
        api_key = api_key,
        api_base= api_base
    )
    return embeddings