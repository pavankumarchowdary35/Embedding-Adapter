import os
from embedding_ada.synthetic_dataset import load_corpus, generate_trian_pairs, EmbeddingQAFinetuneDataset, load_existing_data
from embedding_ada.embedding_model import resolve_embed_model
from embedding_ada.fine_tune_engine import EmbeddingAdapterFinetuneEngine
from embedding_ada.adapter_embedding import AdapterEmbeddingModel
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from embedding_ada.two_layer_cnn import TwoLayerNN 

class EmbeddingModelTrainer:
    def __init__(self, embedding_model_name: str, train_files: list, model_output_path: str = "model_output_test", epochs: int = 10, llm_type: str = "openai", adapter_model=None):
        self.embedding_model_name = embedding_model_name
        self.train_files = train_files
        self.model_output_path = model_output_path
        self.epochs = epochs
        self.llm_type = llm_type.lower()

        # Set the default adapter model if not provided
        if adapter_model is None:
            self.adapter_model = TwoLayerNN(
                384,  # input dimension
                1024,  # hidden dimension
                384,  # output dimension
                bias=True,
                add_residual=True,
            )
        else:
            self.adapter_model = adapter_model

        # Initialize the base embedding model
        self.base_embed_model = resolve_embed_model(self.embedding_model_name)

        # Initialize the LLM based on the llm_type
        if self.llm_type == "azure":
            self.llm_ = AzureOpenAI(
                engine=os.getenv("AZURE_OPENAI_ENGINE", "gpt4"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
            )
        elif self.llm_type == "openai":
            # Make sure the OpenAI API key is set
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set when using OpenAI.")
            
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.llm_ = OpenAI()
        else:
            raise ValueError("Invalid LLM_TYPE argument. Choose either 'azure' or 'openai'.")

    def load_and_generate_dataset(self, verbose: bool = True):
        """Load corpus and generate training pairs."""
        self.train_nodes = load_corpus(self.train_files, verbose=verbose)
        self.train_dataset = generate_trian_pairs(self.train_nodes, self.llm_)

    def fit(self):
        """Fine-tune the embedding model using the adapter."""
        self.finetune_engine = EmbeddingAdapterFinetuneEngine(
            self.train_dataset,
            self.base_embed_model,
            model_output_path=self.model_output_path,
            epochs=self.epochs,
            verbose=True,
            adapter_model=self.adapter_model
        )
        self.finetune_engine.finetune()
        self.embed_model = self.finetune_engine.get_finetuned_model(adapter_cls=TwoLayerNN)

    def transform(self, query: str):
        """Transform a query using the fine-tuned model."""
        if not hasattr(self, 'embed_model'):
            raise ValueError("The model must be fine-tuned before transforming queries.")
        return self.embed_model._get_query_embedding(query)

