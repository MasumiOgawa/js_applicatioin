from flask import Flask, render_template, request, jsonify
from llama_index import download_loader
from pathlib import Path
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
import os
import openai
openai.api_key = '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'
os.environ["OPENAI_API_KEY"] = openai.api_key


app = Flask(__name__)

os.listdir(".")
PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file=Path('The Mafia Animals.pdf'))

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

QA_PROMPT_TMPL = (
    "最大50文字で回答してくださいをしてください。タメ口で女の子らしいけどマフィアらしく回答して下さい。語尾は、「なのだ」 \n"
    "あなたの性格や名前、属性は次のようです。・性別：女の子子・動物：リス・名前：ジゼル・性格：明るい・武器：眼鏡・ファミリー：任侠ファミリー:。"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    " {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
engine = index.as_query_engine(text_qa_template=QA_PROMPT)
index.storage_context.persist()
from llama_index import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./storage/")
index = load_index_from_storage(storage_context)

def generate_response(message):
    response = engine.query(message)
    return response.response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']

    reply = generate_response(message)

    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(debug=True)

