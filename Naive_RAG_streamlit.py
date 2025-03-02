import streamlit as st
from NaiveRAG import NaiveRAG
import os

def main():
    st.title("Naive RAG with Streamlit")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    # Initialize NaiveRAG
    naive_rag = NaiveRAG()
    naive_rag.load_vector_store()
    
    # Check if vector store exists
    if not os.path.exists(naive_rag.vector_store_dir):
        st.write("Creating vector store...")
        naive_rag.load_vector_store()
        document_ids = naive_rag.add_documents_to_vector_store()
        st.write(f"Vector store created with {len(document_ids)} documents.")
    else:
        # st.write("Vector store already exists.")
        pass
    


    # Accept user input
    if query := st.chat_input("Enter your query:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            answer = naive_rag.answer_query(query)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.markdown(answer)

if __name__ == "__main__":
    main()
