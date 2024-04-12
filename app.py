from flask import Flask, render_template, request, jsonify, session,redirect, url_for
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage,MessageRole
from llama_index.llms.together import TogetherLLM
import os
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
)
import re
from llama_index.core.query_engine import RetrieverQueryEngine
import PyPDF2
import shutil
from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext

app = Flask(__name__, static_folder='static',template_folder='templates')

def split_pdf_by_page(pdf_file, output_folder):
    # Delete the output folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # Create the output folder
    os.makedirs(output_folder)

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    pdf_names = []

    for page_num in range(num_pages):
        pdf_writer = PyPDF2.PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_num])
        output_file_path = os.path.join(output_folder, f'page_{page_num + 1}.pdf')
        pdf_names.append(f'page_{page_num + 1}')
        with open(output_file_path, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)

    return pdf_names


def build_document_summary_index(city_docs):
    # Initialize the embedding model
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
    Settings.embed_model = embed_model
    llm = TogetherLLM(
                    model="togethercomputer/llama-2-70b-chat", api_key="1cecf43792b1187b044ab0293853353cc45caf8fdfe0a82e049d53cbf5954d26"
                )
    Settings.llm = llm

    # Set context window size
    Settings.context_window = 4096

    # Initialize the splitter
    splitter = SentenceSplitter(chunk_size=1024)

    # Initialize the response synthesizer
    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", use_async=True)

    # Build the document summary index
    doc_summary_index = DocumentSummaryIndex.from_documents(
        city_docs,
        llm=llm,
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True
    )

    return doc_summary_index



doc_summary_index = None
city_docs = []

def formatted_answer(answer):
    lines = answer.split('\n')
    formatted_lines = []
    in_list = False
    list_type = None

    for line in lines:
        # Check for numbered list
        numbered_match = re.match(r'^(\d+\.\s)(.+)', line)
        # Check for asterisk list
        asterisk_match = re.match(r'^(\*\s)(.+)', line)
        # Split asterisk list items that are on the same line
        asterisk_items = re.findall(r'\*\s(.+?)(?=(\*\s|$))', line)

        if numbered_match:
            if not in_list or list_type != 'ol':
                if in_list:  # Close the previous list
                    formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
                formatted_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            formatted_lines.append(f'<li>{numbered_match.group(2).strip()}</li>')

        elif asterisk_match or asterisk_items:
            if not in_list or list_type != 'ul':
                if in_list:  # Close the previous list
                    formatted_lines.append('</ol>' if list_type == 'ol' else '</ul>')
                formatted_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            if asterisk_items:
                for item, _ in asterisk_items:
                    formatted_lines.append(f'<li>{item.strip()}</li>')
            else:
                formatted_lines.append(f'<li>{asterisk_match.group(2).strip()}</li>')

        else:
            if in_list:  # Close the previous list
                formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')
                in_list = False
            # Wrap non-list lines in paragraphs or handle them appropriately
            formatted_lines.append(f'<p>{line.strip()}</p>')

    # Close any open list tags
    if in_list:
        formatted_lines.append('</ul>' if list_type == 'ul' else '</ol>')

    # Combine all formatted lines
    formatted_output = ''.join(formatted_lines)

    return formatted_output

query_history=[]

from flask import request, render_template, jsonify

@app.route('/process_file', methods=['POST'])
def process_file():
    global doc_summary_index
    global city_docs
    global query_history

    if 'file' in request.files:
        if request.files['file']:
            # Handle PDF file upload
            pdf_file = request.files['file']
            output_folder = 'output_folder'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            city_docs = []
            output_folder = 'output_folder'
            pdf_names = split_pdf_by_page(pdf_file, output_folder)
            for wiki_title in pdf_names:
                docs = SimpleDirectoryReader(
                    input_files=[f"{output_folder}/{wiki_title}.pdf"]
                ).load_data()
                for doc in docs:
                    doc.doc_id = wiki_title
                city_docs.extend(docs)
            doc_summary_index = build_document_summary_index(city_docs)
            doc_summary_index.storage_context.persist("index")
            return jsonify({'status': 'success'})
        else:
            return "No selected file", 400

    elif 'user_input' in request.form:
            user_input = request.form.get('user_input')
            if user_input:
                    retriever = DocumentSummaryIndexLLMRetriever(
                        doc_summary_index,
                        choice_top_k=3,
                    )
                    response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")

                    chat_text_qa_msgs = [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=(
                                "Summarize the documents.\n"
                                "Always answer the query using the provided context information, "
                            ),
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=(
                                "Context information is below.\n"
                                "---------------------\n"
                                "{context_str}\n"
                                "---------------------\n"
                                # "Given the context information, Provide a summary to the document .\n"
                                " Please write a passage to answer the question\n"
                                "Try to include as many key details as possible.\n"


                            ),
                        ),
                    ]

                    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
                    query_2 = f"""{user_input} """
                    query_engine = RetrieverQueryEngine(
                        retriever=retriever,
                        response_synthesizer=response_synthesizer
                    )
                    response__2 = doc_summary_index.as_query_engine(text_qa_template=text_qa_template).query(query_2)
                    answer = formatted_answer(str(response__2))
                    query_history.append({"question": user_input, "answer": answer})
                    return render_template('doc_chat.html', query_history=query_history)
            else:
                        return "No user input provided", 400

@app.route('/')
def index():
    return render_template('index.html',)
@app.route('/doc')
def doc():
    return render_template('doc.html')
@app.route('/doc_chat')
def doc_chat():
    return render_template('doc_chat.html')
@app.route('/athena_chat')
def athena_chat():
    return render_template('athenachat.html')

if __name__ == '__main__':
    app.run(port = 7000, debug=False)