import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import nltk
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import dotenv_values
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage


app = Flask(__name__)
load_dotenv()

config = dotenv_values(".env")
openai_api_key = config["OPENAI_API_KEY"]


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return pages

def chat_model():
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        model='gpt-3.5-turbo-0125'
    )
    return chat


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def chat_bot(pages, query_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    text_content = [doc.page_content for doc in splits]
    docsearch = FAISS.from_texts(text_content, embeddings)
    retriever = docsearch.as_retriever()
    qa_system_prompt = """You are an expert for question-answering tasks. Use the following pieces of retrieved 
    context to answer the question. If the answer is present in the context, please answer exactly same as context. 
    you are also allowed to answer from chat history. If you don't know answer for any question, do not say 'I don't 
    know'. Instead you should say: "Unfortunately, I am unable to answer your question at the moment. Please contact 
    info@nikles.com so that our customer service can provide you with optimal assistance.

    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat_model(), qa_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    chat_history = []

    question = "can you give me contact number?"
    answer = rag_chain.invoke({"input": query_data, "chat_history": chat_history})

    return answer["answer"]


@app.route('/', methods=['POST'])
def index():
    if request.method == "POST":

        json_data = request.json
        if json_data is None or 'question' not in json_data:
            return jsonify({"error": "Question field is missing."}), 400

        query_data = request.json['question']
        pages = load_pdf('pdf/Nikles Company Information Chatbot.pdf')
        ai_message = chat_bot(pages, query_data)
        context = {
            'AL Message': ai_message
        }
        return jsonify(context)

    return jsonify()


if __name__ == '__main__':
    app.run(debug=True)
