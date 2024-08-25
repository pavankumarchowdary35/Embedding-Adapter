import os
from typing import TYPE_CHECKING, List, Optional, Union


from llama_index.core.bridge.langchain import Embeddings as LCEmbeddings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.utils import get_cache_dir


EmbedType = Union[BaseEmbedding, "LCEmbeddings", str]


def resolve_embed_model(embed_model: Optional[EmbedType] = None,
                        callback_manager: Optional[CallbackManager] = None,
                        ) -> BaseEmbedding:
    from llama_index.core.settings import Settings

    try:
        from llama_index.core.bridge.langchain import Embeddings as LCEmbeddings
    except ImportError:
        LCEmbeddings = None

    if embed_model == 'default':
        if os.getenv('IS_TESTING'):
            embed_model = MockEmbedding(embed_dim=8)
            embed_model.callback_manager = callback_manager or Settings.callback_manager
            return embed_model
        try:
            from llama_index.embeddings.openai import (
                OpenAIEmbedding,
            )
            from llama_index.embeddings.openai.utils import (
                validate_openai_api_key,
            )

            embed_model = OpenAIEmbedding()
            validate_openai_api_key(embed_model.api_key)
        except ImportError:
            raise ImportError(
            "`llama-index-embeddings-openai` package not found, "
            "please run `pip install llama-index-embeddings-openai`" 
            )
        except ValueError as e:
            raise ValueError(
                "\n******\n"
                "Could not load OpenAI embedding model. "
                "If you intended to use OpenAI, please check your OPENAI_API_KEY.\n"
                "Original error:\n"
                f"{e!s}"
                "\n******"
            )
        
    if isinstance(embed_model, str):
        try:
            from llama_index.embeddings.huggingface import (
                HuggingFaceEmbedding,
            )
            cache_folder = os.path.join(get_cache_dir(),'models')
            os.makedirs(cache_folder,exist_ok=True)

            embed_model = HuggingFaceEmbedding(model_name= embed_model, cache_folder= cache_folder)

        except ImportError:
            raise ImportError(
                "`llama-index-embeddings-huggingface` package not found, "
                "please run `pip install llama-index-embeddings-huggingface`"
            )
        
    if LCEmbeddings is not None and isinstance(embed_model, LCEmbeddings):
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            from llama_index.embeddings.langchain import(
                LangchainEmbedding,            
            )
            lc_embed_model = HuggingFaceEmbeddings(model_name=embed_model)
            embed_model = LangchainEmbedding(lc_embed_model)
        except ImportError as e:
            raise ImportError(
                "`llama-index-embeddings-langchain` package not found, "
                "please run `pip install llama-index-embeddings-langchain`"
            )
    if embed_model is None:
        print("Embeddings have been explicitly disabled. Using MockEmbedding.")
        embed_model = MockEmbedding(embed_dim=1)
        
    embed_model.callback_manager = callback_manager or Settings.callback_manager

    return embed_model






