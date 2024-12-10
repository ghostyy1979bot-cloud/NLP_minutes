import os
import json
import uuid
import re  # Regular expressions
import threading
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import tiktoken
from flask import send_from_directory

# Suppress all logs below WARNING globally (should come first)
logging.basicConfig(level=logging.WARNING)

# Suppress specific loggers
logging.getLogger('pdfminer').setLevel(logging.WARNING)  # Reduce verbosity of pdfminer logs
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Suppress Werkzeug logs
logging.getLogger('chromadb').setLevel(logging.WARNING)

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

# Configure ChromaDB with Persistent Storage
storage_path = os.path.join(os.getcwd(), "chroma_storage")  # Define a folder for persistent storage
os.makedirs(storage_path, exist_ok=True)  # Ensure the storage path exists

# Initialize ChromaDB client with persistent storage
client = chromadb.PersistentClient(path=storage_path)

# Create or get the collection in ChromaDB
collection = client.get_or_create_collection(name='meeting_minutes')

# Allowed tags and impact options
TAGS = [
    "Management", "Budget", "Curriculum", "Policy", "Research",
    "Student Affairs", "Infrastructure", "Human Resources", "Events",
    "Strategic Planning"
]
IMPACT_OPTIONS = ["Positive", "Negative", "Neutral"]

# Initialize progress dictionary
progress_dict = {}

def fetch_all_records():
    # Query the collection for all records
    results = collection.get(include=["metadatas", "documents"])  # Removed "ids"
    data = []
    for idx, metadata in enumerate(results["metadatas"]):
        record_id = results["ids"][idx]  # Access "ids" directly from the results
        record = {
            "index": idx + 1,  # Sequential number starting from 1
            "id": record_id,   # Include the record ID for deletion
            "metadata": metadata,
            "similarity": None  # No similarity score when fetching all records
        }
        data.append(record)
    return data



def search_records(query):
    # Encode the query using SentenceTransformer
    query_embedding = embedding_model.encode(query).tolist()

    # Get the total number of elements in the collection
    total_elements = collection.count()  # Use `count()` to get the number of elements

    # Set n_results to the minimum of 100 or the total number of elements
    n_results = min(100, total_elements)

    # Perform similarity search using ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,  # Adjusted dynamically
        include=["metadatas", "documents", "distances"]
    )

    data = []
    for idx, (metadata, document, distance) in enumerate(zip(results["metadatas"][0], results["documents"][0], results["distances"][0])):
        similarity = 1 - distance  # Assuming distance is normalized between 0 and 1
        record = {
            "index": idx + 1,
            "metadata": metadata,
            "similarity": similarity  # Store similarity score
        }
        data.append(record)
    return data


# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_meeting_header(content):
    # List of predefined committee names
    predefined_committees = ["Management", "Council", "Postgraduate"]

    prompt_template = f"""
    You are an assistant tasked with extracting key information from meeting headers.

    The meeting header may be in any language. Translate all extracted information into English.

    Extract and format the key information from the following meeting header in JSON format.

    Ensure the output is valid JSON without any additional commentary or explanations.

    **Important:** Pay close attention to the title, especially the "Meeting_Number". Include all parts of the meeting number exactly as they appear, including any special characters (e.g., "@3r").

    If the committee name matches one of the following options, ensure that only the exact match is returned:
    {', '.join(predefined_committees)}

    **Date and Time:** Ensure that the "Date" is in the format "YYYY-MM-DD" and the "Time" is in the 24-hour format "HH:MM".

    Meeting Header:
    {{element}}

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

    # Handle JSON parsing
    try:
        extracted_data = json.loads(result)

        # Validate Committee Name
        committee_name = extracted_data.get("Committee_Name", "").strip()
        if committee_name not in predefined_committees:
            extracted_data["Committee_Name"] = "Unknown"  # Set a default value if no match

        return extracted_data
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from extracted meeting header.")

def num_tokens(text, model="gpt-3.5-turbo"):
    """Estimate the number of tokens in a given text for a specific model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_text_into_chunks(text, max_tokens=4000):
    """
    Splits text into chunks based on approximate token count.
    This function splits by words to avoid breaking sentences.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        # Approximate token length by word length + 1 (for space)
        current_length += len(word) + 1
        if current_length > max_tokens:
            # Add the current chunk to chunks
            chunks.append(" ".join(current_chunk))
            # Reset current_chunk and current_length
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)

    if current_chunk:  # Add any remaining words as the last chunk
        chunks.append(" ".join(current_chunk))

    return chunks

def summarize_large_text(text, model="gpt-3.5-turbo"):
    """
    Summarize large texts by splitting them into smaller chunks,
    summarizing each chunk, and then summarizing the combined summaries.
    """
    # Define chunk size and overlap
    chunk_size = 3500  # Tokens per chunk (adjust as needed)
    chunk_overlap = 200  # Tokens to overlap between chunks

    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    # Split the text into chunks
    chunks = splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")

    summaries = []

    # Summarize each chunk
    for idx, chunk in enumerate(chunks, 1):
        print(f"Summarizing chunk {idx}/{len(chunks)}...")
        try:
            summary_data = summarize_text(chunk, model=model)
            summaries.append(summary_data.get('summary', ''))
        except Exception as e:
            print(f"Error summarizing chunk {idx}: {e}")
            summaries.append('')  # Append empty string or handle as needed

    # Combine all summaries into a single text
    combined_summaries = "\n".join(summaries)
    print("Summarizing combined summaries...")

    # Summarize the combined summaries
    final_summary = summarize_text(combined_summaries, model=model)
    return final_summary

def summarize_text(text, model="gpt-3.5-turbo"):
    """
    Summarize the given text using the specified language model.
    If the text exceeds the model's token limit, it will be split and summarized.
    """
    # Estimate token count
    tokens = num_tokens(text, model=model)
    max_tokens = 4000  # Adjust based on model's limit

    if tokens > max_tokens:
        print("Text exceeds token limit. Initiating chunked summarization.")
        return summarize_large_text(text, model=model)

    prompt_template = """
    You are an assistant tasked with summarizing meeting minutes.

    Summarize the meeting minute text in plain English, providing detailed information about each point, including specific prices, quantities, and current statuses of equipment or projects, such as the chiller's price and procurement progress. Include any relevant dates, departments involved, and additional context provided. Put everything in form of paragraph

    Additionally, assign an appropriate tag and impact from the provided lists.

    Tags (choose one that best fits):
    - Management
    - Budget
    - Curriculum
    - Policy
    - Research
    - Student Affairs
    - Infrastructure
    - Human Resources
    - Events
    - Strategic Planning

    Based on the meeting summary, determine the impact (choose one: Positive, Negative, Neutral). 
    Criteria:
    - Positive: Includes resolved issues, improvements, or beneficial actions.
    - Negative: Includes unresolved challenges, inefficiencies, or adverse effects.
    - Neutral: Includes routine updates or plans with no notable outcome.
    Consider the following:
    - Does it address an unresolved issue or problem? (Negative)
    - Does it improve current processes or outcomes? (Positive)
    - Is it a routine matter or an update with no clear implications? (Neutral)

    Text:
    {minit_content}

    Your response should be in JSON format without any additional commentary.

    Example format:
    {{
        "summary": "Your summary here.",
        "tag": "Selected tag",
        "impact": "Selected impact"
    }}

    Summary:
    """


    human_message_prompt = HumanMessagePromptTemplate.from_template(prompt_template)
    chat_prompt = ChatPromptTemplate.from_messages([human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt)
    chain_input = {'minit_content': text}
    result = chain.run(chain_input)
    # Handle JSON parsing
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from summarization output.")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Handle PDF upload
        if 'pdf_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['pdf_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Generate a unique ID for this task
            task_id = str(uuid.uuid4())
            # Initialize progress
            progress_dict[task_id] = 'Starting...'

            # Start processing in a background thread
            thread = threading.Thread(target=process_pdf_task, args=(filename, task_id))
            thread.start()

            # Return JSON response with task ID
            return jsonify({'task_id': task_id})
        else:
            return jsonify({'error': 'Invalid file type'}), 400

    # Handle GET requests for search or display all records
    query = request.args.get('query')
    if query:
        # Perform similarity search
        records = search_records(query)
        # Exclude similarity scores before passing to the front-end
        for record in records:
            record.pop('similarity', None)
    else:
        # Fetch all records without similarity scores
        records = fetch_all_records()
    return render_template('upload.html', records=records)

def process_pdf_task(filename, task_id):
    try:
        progress_dict[task_id] = 'Loading PDF content...'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Load the PDF content
        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy="hi_res"
        )
        docs = loader.load()

        if not docs:
            progress_dict[task_id] = 'Error: No content found in the PDF.'
            return

        progress_dict[task_id] = 'Extracting header information...'

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
            header_info['Committee_Name'] = header_info.get('Committee_Name', 'N/A')  # Default to 'N/A' if not found
        except Exception as e:
            progress_dict[task_id] = f'Error extracting header information: {e}'
            return

        progress_dict[task_id] = 'Extracting meeting minutes...'

        # Extract meeting minutes
        minit_contents = {}
        current_minit = None
        
        minit_pattern = re.compile(r'^min(?:\.|it)?\s*(\d+)\s*:', re.IGNORECASE)
        end_marker_pattern = re.compile(r"disemak\s*dan\s*disahkan", re.IGNORECASE)  # Pattern for "Disemak dan disahkan", ignoring spaces and case

        for doc in docs:
            page_number = doc.metadata.get("page_number", 1)
            if page_number >= first_page_number:
                text = doc.page_content.strip()
                match = minit_pattern.match(text)
                if match:
                    current_minit = match.group(1)  # Capture only the numeric value
                    minit_contents[current_minit] = text
                    print(f"Started new minute: {current_minit}")
                elif current_minit:
                    minit_contents[current_minit] += "\n" + text
                    print(f"Appended text to minute {current_minit}")

                # Check for the end marker
                if end_marker_pattern.search(text):
                    print(f"End marker detected on page {page_number}. Last minute identified: {current_minit}")
                    break  # Stop processing further as the last minute is identified


        # Debug: Print all collected minutes
        print(f"Collected minutes: {list(minit_contents.keys())}")

        progress_dict[task_id] = 'Summarizing minutes...'

        # Summarize each minute
        minit_summaries = {}
        for minit_number, content in minit_contents.items():
            try:
                summary_data = summarize_text(content)
                minit_summaries[minit_number] = {
                    'content': content,
                    'summary': summary_data.get('summary', ''),
                    'tag': summary_data.get('tag', ''),
                    'impact': summary_data.get('impact', '')
                }
                print(f"Summarized minute {minit_number}")
            except Exception as e:
                minit_summaries[minit_number] = {
                    'content': content,
                    'summary': 'Error summarizing content.',
                    'tag': '',
                    'impact': ''
                }
                print(f"Error summarizing minute {minit_number}: {e}")

        # Save results in progress_dict without creating an overall summary
        progress_dict[task_id] = {
            'status': 'Done',
            'header_info': header_info,
            'minit_summaries': minit_summaries,
            'filename': filename
        }
    except Exception as e:
        progress_dict[task_id] = f'Error: {e}'

@app.route('/progress/<task_id>')
def progress(task_id):
    progress = progress_dict.get(task_id, 'Starting...')
    if isinstance(progress, dict) and progress.get('status') == 'Done':
        # Return a flag indicating that processing is complete
        return jsonify({'progress': 'Done', 'complete': True})
    elif isinstance(progress, str) and progress.startswith('Error'):
        return jsonify({'progress': progress, 'complete': True})
    else:
        return jsonify({'progress': progress, 'complete': False})

@app.route('/edit_info/<task_id>', methods=['GET', 'POST'])
def edit_info(task_id):
    data = progress_dict.get(task_id)
    if not data or not isinstance(data, dict) or data.get('status') != 'Done':
        flash('Processing not complete or task not found.')
        return redirect(url_for('upload_file'))

    header_info = data['header_info']
    minit_summaries = data['minit_summaries']
    filename = data['filename']

    return render_template(
        'edit_info.html',
        header_info=header_info,
        minit_summaries=minit_summaries,
        tags=TAGS,
        impact_options=IMPACT_OPTIONS,
        filename=filename
    )

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/delete_record/<record_id>', methods=['POST'])
def delete_record(record_id):
    try:
        # Delete the record from the ChromaDB collection
        collection.delete(ids=[record_id])
        return jsonify({'success': True, 'message': 'Record deleted successfully.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error deleting record: {e}'})


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
    minit_numbers = request.form.getlist('minit_number[]')
    summaries = request.form.getlist('summary[]')
    tags = request.form.getlist('tag[]')
    impacts = request.form.getlist('impact[]')
    pdf_name = request.form.get('filename')

    for idx, minit_number in enumerate(minit_numbers):
        summary = summaries[idx]
        tag = tags[idx]
        impact = impacts[idx]

        # Compute embedding
        embedding = embedding_model.encode(summary).tolist()

        # Metadata for the record
        metadata = {
            'pdf_name': pdf_name,
            'committee_name': header_info['Committee_Name'],
            'minit_number': minit_number,
            'summary': summary,
            'tag': tag,
            'date': header_info.get('Date', ''),
            'impact': impact,
        }

        # Check for existing record using `$and` in the `where` clause
        try:
            existing_records = collection.get(
                where={"$and": [
                    {"pdf_name": {"$eq": pdf_name}},
                    {"minit_number": {"$eq": minit_number}}
                ]},
                include=["metadatas", "documents"]  # Removed "ids" from include
            )

            if existing_records["metadatas"]:
                # Since "ids" are not included, but "collection.get()" returns "ids" by default,
                # we can access them directly from the result.
                record_id = existing_records["ids"][0]
                collection.update(
                    ids=[record_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
            else:
                # Add a new record
                doc_id = str(uuid.uuid4())
                collection.add(
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
        except Exception as e:
            print(f"Error saving data: {e}")
            flash(f"Error saving data: {e}", "danger")
            continue

    flash('Data saved successfully.')
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(port=5001, debug=True)