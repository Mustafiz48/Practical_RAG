from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_chroma import Chroma
import getpass
import os
from dotenv import load_dotenv

#get api key from .env file
load_dotenv()
google_api_key = os.getenv("google_api_key")
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass(google_api_key)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

file_path = "History_of_Bangladesh.pdf"

# Load the PDF file
loader = PyPDFLoader(file_path)
pages = []
i=0
for page in loader.lazy_load():
        i+=1
        # print(f"Loaded page: {i}")
        pages.append(page)

print(f"Total pages: {len(pages)}")
print(f"First page: {pages[3].page_content[:100]}")

print()
print()

# split the document
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index = True,
)

all_splits = text_splitter.split_documents(documents=pages)
print(f"Total splits: {len(all_splits)}")
print(f"First split: {all_splits[0]}")
print()
print()

# setup embedder
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

print(embeddings.embed_query("Hello World!"))

print()
print()

# set up vector store
vectore_store = Chroma(embedding_function=embeddings)

# store the documents into vectore store

document_ids = vectore_store.add_documents(all_splits)

print(document_ids[:3])


# Retrieve the most similar documents

def retrieve_docs(query):
    retrieved_docs = vectore_store.similarity_search(query)
    return retrieved_docs

def answer_query():
    query = input("Enter your query: ")
    retrieved_docs = retrieve_docs(query)
    print(f"Retrieved {len(retrieved_docs)} documents")
    
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i}: {doc.page_content}")

    content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    print(content[:100])

    message = f"""
                You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
                Question: {query} 
                Context: {content} 
                Answer:
                """
    answer = llm.invoke(message)
    return answer.content

answer = answer_query()
print()
print()

print(f"Answer: {answer}")