# 🎉 embedding-ada

## 📜 Description

embedding-ada is a Python package used to fine-tune embedding adapter on top of embedding models from Hugging Face and OpenAI. This package is build using Llama-Index.

## Why Use embedding-ada?
Improved Retrieval Performance: By fine-tuning adapters, this package helps you bring relevant documents closer to the query embeddings, improving the results of approximate nearest neighbor searches.

## 🚀 Installation
You can install **embedding-ada** directly from PyPI using pip:

```bash
pip install embedding-ada
```
## 💻 Usage
```python
from embedding_ada.adapter import EmbeddingModelTrainer
# Define your training files
train_files = ["path/to/your/example.pdf"]

# Initialize the trainer
trainer = EmbeddingModelTrainer(
    embedding_model_name='BAAI/bge-small-en',
    train_files=train_files,
    model_output_path="model_output_test",
    epochs=10,
    llm_type="openai"  # or "azure" if using Azure OpenAI
)
# Load train files and generate dataset out of it.
trainer.load_and_generate_dataset(verbose=True)

#Fine-tune the model using the dataset
trainer.fit()

#Finally, transform a query using the fine-tuned model
transformed_query_embedding = trainer.transform('query_text')  

```
## Overview

The user has to provide text on which he want to trian the adapter model. The text can be in either .pdf or .txt files. This text is parsed into nodes, with each node representing a "chunk" of a source document. Using a language model (LLM), we generate a question from each node. This process forms query-context pairs, which are then used to train a neural network model. The default model is a simple two-layer neural network with ReLU activation and a residual layer at the end. Users also have the option to define a custom neural network as the adapter model and train it with the formulated query-context pairs.

## Training Process

The training process uses the MultipleNegativesRankingLoss function, similar to the one used in training sentence_transformers. This loss function is particularly effective for training embeddings in retrieval setups where you have positive pairs (e.g., query and relevant document). During training, for each batch, the function randomly samples n-1 negative documents, making it robust for retrieval tasks.

## Embedding Adapters

The core concept behind embedding adapters is as follows: Given a set of query embeddings, and corresponding sets of relevant and irrelevant document embeddings for each query, the adapter model learns a transformation that adjusts the vector space. This transformation "squeezes" and "rotates" the space, mapping it to a new space where relevant documents are closer to the query. We refer to this transformation as an 'adapter,' as it is applied after the output of the embedding model.

## RAG Use Case

Consider a scenario where you have a large corpus of text, and you want to build a Q&A system on that corpus. The initial step involves splitting the text corpus into chunks, converting these chunks into embeddings, and storing them in a vector database.

When a user makes a query, the goal is to retrieve text chunks from the vector database that are relevant to the query. Vector databases utilize Approximate Nearest Neighbors (ANN) algorithms, such as HNSW, to fetch approximate nearest neighbors to the query embedding.

Instead of performing an ANN search with the original query, you perform the search with the transformed query (transformed using the trained adapter model). The results are significantly better because the transformed query embeddings are intended to be closer to the relevant text chunks.


# 📄License
This project is licensed under the MIT License.

# 🛠️ Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.

# 📬 Contact
If you have any questions, feel free to reach out:
Email: pavankumarchowdary35@gmail.com
GitHub: pavankumarchowdary35

**🔔 Note:**

To generate a synthetic dataset from the user-provided training files, you need to configure an LLM instance.

### 🌐 Using OpenAI

If you want to use the OpenAI instance, please set your OpenAI API key:

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
```

If you want to use Azure OpenAI, set the following variables
```python
import os
os.environ["LLM_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENGINE"] = "gpt4"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-azure-endpoint.com"
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_OPENAI_API_VERSION"] = "Version"
```