import pdfplumber
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


@app.route('/', methods=['POST'])
def index():
    if request.method == "POST":
        query_data = request.args.get('question')
        pdf_text = extract_text_from_pdf('pdf/Nikles Company Information Chatbot.pdf')
        return pdf_text
    return jsonify()


if __name__ == '__main__':
    app.run(debug=True)
