# ChatPDF

Talking with a PDF using ChatPDF

## What is ChatPDF

ChatPDF is a minimumist python application used to perform queries against a PDf file with ChatPGT using [embeddings](https://platform.openai.com/docs/guides/embeddings) and [langchain](https://github.com/hwchase17/langchain).

## Why ChatPDF

ChatGPT is trained with internet data, which works well with generalist content, however, it is not optimised for specific content you are interested in, such as PDFs. This project aims to demo this ability using the concept of embedding.

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
