import dotenv
import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

uploaded_file = st.file_uploader("Choose a PDF file", type = "pdf")
if uploaded_file is not None:
    # load PDf
    with st.spinner("Loading PDf..."):
        text = read_pdf(uploaded_file)
    st.success("Finished loading PDF.")

    # chunking PDF
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 2000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    # question
    query = st.text_input("Your question:")
    response = ""
    if query:
        with st.spinner("Running query..."):
            docs = knowledge_base.similarity_search(query)
            
            # print(str(docs))
            # print(str(query))

            llm = OpenAI(openai_api_key = openai_api_key, temperature = 0.7)
            chain = load_qa_chain(llm, chain_type = "stuff", prompt = PROMPT)
            response = chain.run(input_documents = docs, question = query)

        st.success("Completed query.")
        st.write("Answer: ", response)