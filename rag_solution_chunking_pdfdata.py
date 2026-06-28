# %%
pip install langchain_community

# %%
import os
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_text_splitters import MarkdownHeaderTextSplitter

# %%
# Step 1: Loading a File
from langchain_community.document_loaders import TextLoader
loader = TextLoader("/content/sample_data/tennis_details.md")
text_doc = loader.load()
#print(text_doc[0].page_content)

# %%
# Step 2: divide the data into chunks
from langchain_text_splitters import MarkdownHeaderTextSplitter
split_condition = [("##", "title")]
splitter = MarkdownHeaderTextSplitter(split_condition)
doc_splits = splitter.split_text(text_doc[0].page_content)
#print(doc_splits)
text_chunks = [split.page_content for split in doc_splits]
print(text_chunks)

# %%
print(len(text_chunks))

# %%
# Step 3: Generate Embeddings

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunk(chunk):
  return embedding_model.encode([chunk], normalize_embeddings = True)

# %%
sample_embedding = embed_chunk(text_chunks[1]).tolist()[0]

# %%
print(sample_embedding)

# %%
len(sample_embedding)

# %%
pip install chromadb

# %%
# Step 4: Store embeddings in ChromaDB

vector_db = Chroma.from_texts(text_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), persist_directory="/tmp/chroma_db")


# %%
vector_db._collection.get(include=['embeddings','documents'])

# %%
#step 5: Set up a LLM
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

# %%
# Step 6: Retrieval and Generation
def retrieve_and_generate(query, threshold=1):
    """Retrieves relevant context from the vector database and generates an answer."""
    search_results = vector_db.similarity_search_with_score(query, k=1)

    print(search_results)

    if not search_results or search_results[0][1] > threshold:
        return "I don't know the answer. There is no available context in vector DB."

    retrieved_context = search_results[0][0].page_content
    similarity_score = search_results[0][1]
    print(f"Similarity Score: {similarity_score}")
    print(f"Retrieved Context: {retrieved_context}")

    prompt = f"Answer the question using the given context\nContext: {retrieved_context}\nQuestion: {query}\nAnswer: "
    print(prompt)
    response = pipe(prompt, max_new_tokens=100)
    return response[0]["generated_text"]

# %%
question = "what is tennis"
response = retrieve_and_generate(question)
print(response)

# %%

