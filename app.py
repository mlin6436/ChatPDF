import os
# from dotenv import load_dotenv, find_dotenv ### For Local
import tempfile
import streamlit as st
from streamlit_chat import message as chat
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

class Chain:
    def __init__(self, openai_api_key):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.knowledge_base = None
        self.qa_chain = None
        self.chat_history = []

    def upload_file(self, file_path):
        loader = PyPDFLoader(file_path=file_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        if self.knowledge_base is None:
            self.knowledge_base = FAISS.from_documents(chunks, self.embeddings)
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, 
                retriever=self.knowledge_base.as_retriever()
            )
            self.chat_history = []
        else:
            self.knowledge_base.add_documents(chunks)

def reset_sessions():
    st.session_state["spinner_uploading"] = st.empty()
    st.session_state["spinner_querying"] = st.empty()

def init_chain():
    ### For Local
    # dotenv = find_dotenv(".env")
    # if dotenv:
    #     load_dotenv()

    # if os.getenv("OPENAI_API_KEY"):
    #     st.session_state["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    # else:
    #     st.text_input(
    #         "OpenAI API Key",
    #         value=st.session_state["OPENAI_API_KEY"], 
    #         key="text_input_openai_api_key",
    #         type="password"
    #     )

    st.session_state["OPENAI_API_KEY"] = ""
    st.session_state["chain"] = None

    st.text_input(
        "OpenAI API Key",
        value=st.session_state["OPENAI_API_KEY"], 
        key="text_input_openai_api_key",
        type="password"
    )

    # TODO: validate api key
    if st.session_state["OPENAI_API_KEY"]:
        st.session_state["chain"] = Chain(st.session_state["OPENAI_API_KEY"])

def upsert_files():
    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["spinner_uploading"], st.spinner(f"Uploading {file.name}"):
            st.session_state["chain"].upload_file(file_path)
        os.remove(file_path)

def qa():
    if st.session_state["text_input_user_query"] is not None:
        user_query = st.session_state["text_input_user_query"].strip()
        if len(user_query) > 0:
            with st.session_state["spinner_querying"], st.spinner("Processing"):
                response = st.session_state["chain"].qa_chain({"question": user_query, "chat_history": st.session_state["chain"].chat_history})
                answer = response["answer"]
                st.session_state["chain"].chat_history.append((user_query, answer))

if __name__ == "__main__":
    st.set_page_config(page_title="Chat PDF")
    st.title("Chat PDF")

    reset_sessions()

    if "OPENAI_API_KEY" not in st.session_state:
        init_chain()

    file = st.file_uploader(
        "Upload a PDF file", 
        key="file_uploader",
        on_change=upsert_files,
        accept_multiple_files=True,
        type=["pdf"]
    )

    if file:
        st.text_input(
            "Ask your question here:",
            key="text_input_user_query",
            on_change=qa
        )

    st.divider()

    st.subheader("Chat History")
    if "chain" in st.session_state and st.session_state["chain"] is not None:
        for user_query, response in st.session_state["chain"].chat_history:
            chat(user_query, is_user=True)
            chat(response, is_user=False)

    st.markdown("[Source Code](https://github.com/mlin6436/chatpdf)")