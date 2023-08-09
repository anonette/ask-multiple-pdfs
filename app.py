import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub




def get_docs_text(pdf_docs):
    text = ""
    #check if txt or pdf
    for doc in pdf_docs:
        doctype = doc.type
        if doctype == "text/plain":
            # append text from text file
            text += doc.read().decode("utf-8")
        elif doctype == "application/pdf":
            pdf_reader = PdfReader(doc)
            for page in range( len(pdf_reader.pages)):
                page_content = pdf_reader.pages[page].extract_text()
                text += page_content
        else:
            print(f"Unsupported file type: {doctype}")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_db(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # FIASS
    db = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # db.save_local("storage\\faiss_db")

    # chromadb
    # db = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory="storage\\chromadb")
    return db

def get_conversation_chain(db):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your documents before asking a question.")
        return
    
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Reverse the chat history in chunks of 2 to keep the Q&A pairs in order
    reversed_chat_history = [st.session_state.chat_history[i:i + 2] for i in range(0, len(st.session_state.chat_history), 2)][::-1]

    for chunk in reversed_chat_history:
        for i, message in enumerate(chunk):
            if i % 2 == 0:
                st.write(
                    user_template.replace("{{MSG}}", message.content).replace("{{IMG}}", IMG), 
                    unsafe_allow_html=True)
            else:
                st.write(
                    bot_template.replace("{{MSG}}", message.content), 
                    unsafe_allow_html=True)

import requests
from PIL import Image
import io

def get_user_image():
    if "user_image_url" not in st.session_state:
        # You can download the image and process it here if needed
        st.session_state.user_image_url = "https://i.pravatar.cc/300"
    return st.session_state.user_image_url


def main():
    load_dotenv()
    global IMG
    IMG = get_user_image()

    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "db" not in st.session_state:
        st.session_state.db = None
    if "IMG" not in st.session_state:
        st.session_state.IMG = None

        
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF/TXTs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_docs_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                db = get_db(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    db)


if __name__ == '__main__':
    main()
