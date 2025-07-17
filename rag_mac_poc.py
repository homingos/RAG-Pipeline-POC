import os
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


STRICT_PROMPT_TEMPLATE = """
You are a highly specialized question-answering assistant for an employee policy document. Your process must be followed exactly.

Here is the context from the document:
---
{context}
---

Follow these steps to generate your answer:
1. First, carefully analyze the user's question: "{question}".
2. Second, examine the provided context. Does the context contain a direct answer to the question?
3. If the answer is in the context, provide a concise answer based ONLY on that context.
4. If the answer is NOT in the context, you MUST respond with the exact phrase: "I cannot answer this question as the information is not found in the provided document."

Do not use any external knowledge. Do not add any extra information.

Final Answer:
"""


FLEXIBLE_PROMPT_TEMPLATE = """
You are a helpful assistant. Use the following context to answer the question.
If the context is not relevant or does not contain the answer, use your own knowledge to answer, but mention that the information was not in the provided document.

Context:
{context}

Question:
{question}

Answer:
"""


def main():
    
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        print("Please create a 'config.json' file.")
        return
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    
    pdf_path = config["pdf_path"]
    model_path = config["model_path"]
    retrieval_params = config["retrieval_params"]
    generation_params = config["generation_params"]

    print("--- Configuration Loaded ---")
    print(f"PDF Path: {pdf_path}")
    print(f"Model Path: {model_path}")
    print(f"Strict Prompt Mode: {generation_params['use_strict_prompt']}")
    print(f"Temperature: {generation_params['temperature']}")
    print("--------------------------\n")


    
    if not os.path.exists(pdf_path) or not os.path.exists(model_path):
        print("Error: PDF or Model file path in config.json is invalid.")
        return

   
    print(f"Loading PDF: '{os.path.basename(pdf_path)}'...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=retrieval_params["chunk_size"],
        chunk_overlap=retrieval_params["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    
    print("Creating local embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

   
    print("Loading local LLM with Metal support...")
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=4096,
        temperature=generation_params["temperature"],
        max_tokens=generation_params["max_tokens"],
        verbose=False,
    )

    
    if generation_params["use_strict_prompt"]:
        template = STRICT_PROMPT_TEMPLATE
    else:
        template = FLEXIBLE_PROMPT_TEMPLATE
    
    prompt = ChatPromptTemplate.from_template(template)

   
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    
    print("\n\n--- Local RAG System Ready ---")
    while True:
        user_question = input("\nAsk a question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        
        print("\nThinking...")
        answer = rag_chain.invoke(user_question)
        print("\nAnswer:", answer.strip())
        print("-" * 50)

if __name__ == "__main__":
    main()