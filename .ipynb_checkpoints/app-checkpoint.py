from flask import Flask, render_template, request, redirect, url_for, flash
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import UnstructuredPDFLoader
from werkzeug.utils import secure_filename
import json
import uuid
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize OpenAI LLM
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key is None:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_api_key
)

# Initialize SentenceTransformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection in ChromaDB
collection = client.create_collection(name='meeting_minutes')

# Allowed tags and impact options
TAGS = [
    "Management", "Budget", "Curriculum", "Policy", "Research",
    "Student Affairs", "Infrastructure", "Human Resources", "Events",
    "Strategic Planning"
]
IMPACT_OPTIONS = ["Positive", "Negative", "Neutral"]

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_meeting_header(content):
    prompt_template = """
    You are an assistant tasked with extracting key information from meeting headers.

    Extract and format the key information from the following meeting header in JSON format.

    Ensure the output is valid JSON without any additional commentary or explanations.

    **Important:** Pay close attention to the title, especially the "Meeting_Number". Include all parts of the meeting number exactly as they appear, including any special characters (e.g., "@3r").

    Meeting Header:
    {element}

    The JSON keys should include:
    - "Meeting_Number" (include all parts of the meeting number as they appear in the title)
    - "Committee_Name"
    - "Date"
    - "Time"
    - "Location"
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    chain_input = {'element': content}
    result = chain.run(chain_input)
    return json.loads(result)

def summarize_text(text):
    prompt_template = """
    You are an assistant tasked with summarizing meeting minutes.
    Provide a concise summary for the following meeting minute text in plain English.

    Text:
    {minit_content}

    Summary:
    """
    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    chain_input = {'minit_content': text}
    summary = chain.run(chain_input)
    return summary.strip()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'pdf_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Process the PDF
            return redirect(url_for('process_pdf', filename=filename))
    return render_template('upload.html')

@app.route('/process_pdf/<filename>', methods=['GET', 'POST'])
def process_pdf(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Load the PDF content
    loader = UnstructuredPDFLoader(
        file_path,
        mode="elements",
        strategy="hi_res"
    )
    docs = loader.load()
    if not docs:
        flash('No content found in the PDF.')
        return redirect(url_for('upload_file'))

    # Extract header information from the first page
    first_page_number = docs[0].metadata.get("page_number", 1)
    first_page_elements = [
        doc for doc in docs
        if doc.metadata.get("page_number", 1) == first_page_number
    ]
    first_page_content = "\n".join(
        element.page_content for element in first_page_elements
    )
    try:
        header_info = extract_meeting_header(first_page_content)
    except Exception as e:
        flash(f'Error extracting header information: {e}')
        return redirect(url_for('upload_file'))

    # Extract meeting minutes (assuming they are after the first page)
    minit_contents = {}
    current_minit = ''
    for doc in docs:
        if doc.metadata.get("page_number", 1) > first_page_number:
            text = doc.page_content.strip()
            if text.startswith("MINIT"):
                current_minit = text.split()[0]
                minit_contents[current_minit] = text
            elif current_minit:
                minit_contents[current_minit] += "\n" + text

    # Summarize each minute
    minit_summaries = {}
    for minit_number, content in minit_contents.items():
        summary = summarize_text(content)
        minit_summaries[minit_number] = {
            'content': content,
            'summary': summary,
            'tag': '',  # Placeholder for tag
            'impact': ''  # Placeholder for impact
        }

    # Render the editable form
    return render_template(
        'edit_info.html',
        header_info=header_info,
        minit_summaries=minit_summaries,
        tags=TAGS,
        impact_options=IMPACT_OPTIONS,
        filename=filename
    )

@app.route('/save_data', methods=['POST'])
def save_data():
    # Get data from form
    header_info = {
        'Meeting_Number': request.form.get('Meeting_Number'),
        'Committee_Name': request.form.get('Committee_Name'),
        'Date': request.form.get('Date'),
        'Time': request.form.get('Time'),
        'Location': request.form.get('Location'),
    }
    minit_numbers = request.form.getlist('minit_number')
    summaries = request.form.getlist('summary')
    tags = request.form.getlist('tag')
    impacts = request.form.getlist('impact')
    pdf_name = request.form.get('filename')

    # Store data in ChromaDB
    for idx, minit_number in enumerate(minit_numbers):
        summary = summaries[idx]
        tag = tags[idx]
        impact = impacts[idx]

        # Compute embedding
        embedding = embedding_model.encode(summary).tolist()

        # Create metadata
        metadata = {
            'pdf_name': pdf_name,
            'minit_number': minit_number,
            'summary': summary,
            'tag': tag,
            'date': header_info.get('Date', ''),
            'impact': impact,
        }

        # Generate a unique ID
        doc_id = str(uuid.uuid4())

        # Store in ChromaDB
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )

    flash('Data saved successfully.')
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(debug=True)
