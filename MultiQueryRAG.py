from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_chroma import Chroma
import getpass
import os
from dotenv import load_dotenv
# Load API key from .env file
load_dotenv()
google_api_key = os.getenv("google_api_key")

class MultiQueryRAG:
    def __init__(self):
        self.google_api_key = google_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=google_api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        self.file_path = "History_of_Bangladesh.pdf"
        self.vector_store_dir = "chroma_db"
        self.vectore_store = None

    def load_pdf(self,):
        loader = PyPDFLoader(self.file_path)
        pages = []
        for page in loader.lazy_load():
                pages.append(page)

        print(f"Total pages: {len(pages)}")
        print(f"First page: {pages[3].page_content[:100]}")
        return pages


    def split_document(self, pages):
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index = True,
                )

        all_splits = text_splitter.split_documents(documents=pages)
        print(f"Total splits: {len(all_splits)}")
        print(f"First split: {all_splits[0]}")
        return all_splits

    def load_vector_store(self):
        if self.vectore_store is None:
            self.vectore_store = Chroma(embedding_function=self.embeddings, persist_directory=self.vector_store_dir)
        return self.vectore_store

    def add_documents_to_vector_store(self, file_path=None):
        if file_path:
            self.file_path = file_path
        pages = self.load_pdf()
        splits = self.split_document(pages)
        self.load_vector_store()
        document_ids = self.vectore_store.add_documents(splits)
        return document_ids
    
    def generate_query(self, query, k =5):
        message =f"""
            You are an AI language model assistant. Your task is to generate {k} 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines. Original question: {query}
            """
        simalr_queries = self.llm.invoke(message).content
        similar_queries = list(map(lambda x: x.strip(), simalr_queries.split("\n")))
        print(f"Generated queries: {similar_queries}")
        return similar_queries

    def retrieve_docs(self, query):
        retrieved_docs = self.vectore_store.similarity_search(query, k=5)
        return retrieved_docs

    def answer_query(self, query=None):
        if query is None:
            query = input("Enter your query: ")
        similar_queries = self.generate_query(query)
        similar_queries.append(query)
        content = ""
        for query in similar_queries:
            retrieved_docs = self.retrieve_docs(query)
            content += "\n\n".join(doc.page_content for doc in retrieved_docs)

        print(f"Retrieved content: {content} \n")
        message = f"""
                    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
                    Question: {query} 
                    Context: {content} 
                    Answer:
                    """
        answer = self.llm.invoke(message)
        return answer.content

if __name__ == "__main__":
    multi_query_rag = MultiQueryRAG()
    # if not os.path.exists(multi_query_rag.vector_store_dir):
    multi_query_rag.load_vector_store()
    # multi_query_rag.add_documents_to_vector_store()
    answer = multi_query_rag.answer_query()
    print("Answer: \n\n")
    print(answer)
