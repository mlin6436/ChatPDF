# ChatPDF

Talking with a PDF using ChatPDF

## What is ChatPDF

ChatPDF is a simple Python application that utilises [embeddings](https://platform.openai.com/docs/guides/embeddings) and [langchain](https://github.com/hwchase17/langchain) to execute queries on PDF files through ChatPGT."

## Why ChatPDF

ChatGPT is excellent at processing general content, however, it is not optimised for specialised content like PDFs. This project aims to showcase how to leverage ChatGPT and embeddings against specific content.

## How to run the app

### Install dependencies
```
pip install -r requirements
```

### Retrieve your OpenAI API key

```
https://platform.openai.com/account/api-keys
```

### Insert OpenAI API key to `.env`

```
echo "OPENAI_API_KEY='your-api-key'" >> ~/.env
```

### Start the application

```
streamlit run app.py
```
