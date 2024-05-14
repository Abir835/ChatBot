import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from dotenv import dotenv_values
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

app = Flask(__name__)
load_dotenv()

config = dotenv_values(".env")
openai_api_key = os.getenv("OPENAI_API_KEY")


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


def split_docs(pages):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(pages)
        return splits
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def prompt_template():
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

    return qa_prompt


def chat_bot(pages, query_data):

    splits = split_docs(pages)
    embeddings = OpenAIEmbeddings()
    text_content = [doc.page_content for doc in splits]
    docsearch = FAISS.from_texts(text_content, embeddings)
    retriever = docsearch.as_retriever()

    question_answer_chain = create_stuff_documents_chain(chat_model(), prompt_template())

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    chat_history = []

    question = "can you give me contact number?"
    answer = rag_chain.invoke({"input": query_data, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), answer["answer"]])

    return answer["answer"]


@app.route("/")
def index():
    return jsonify("Chat Bot App Working")


@app.route('/chat_bot', methods=['POST'])
def chat_response():
    if request.method == "POST":
        json_data = request.json
        if json_data is None or 'question' not in json_data:
            return jsonify({"error": "Question field is missing."}), 400

        if 'chat_bot_name' not in json_data:
            return jsonify({"error": "Chat bot name is missing."}), 400

        query_data = json_data['question']
        pdf_name_search = json_data['chat_bot_name']

        try:
            pages = load_pdf("static/pdf/" + pdf_name_search + ".pdf")
            ai_message = chat_bot(pages, query_data)
            data = {
                'AI Message': ai_message
            }
            return jsonify(data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid request method.'}), 405


@app.route('/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    file_name = request.form.get('file_name')

    if not file_name:
        return jsonify({'error': 'File name is missing.'}), 400

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    app.config['UPLOAD_FOLDER'] = config['UPLOAD_FOLDER']

    try:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name + ".pdf"))
        return jsonify({'message': 'File uploaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
