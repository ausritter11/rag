import os
import streamlit as st
import tempfile
from typing_extensions import TypedDict, List
import io
import fitz  # PyMuPDF for PDF handling
import pandas as pd
import csv

try:
    import docx
except ImportError:
    # Make docx optional
    docx = None

from langchain_community.chat_models import ChatOpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# Set page configuration
st.set_page_config(
    page_title="Hytopia Insights",
    page_icon="🔍",
    layout="wide",
)

# App title and description
st.title("💬 Hytopia Insights")
st.markdown("Upload documents and ask questions to get AI-powered answers!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "graph" not in st.session_state:
    st.session_state.graph = None

if "documents" not in st.session_state:
    st.session_state.documents = []

if "file_processor_ready" not in st.session_state:
    st.session_state.file_processor_ready = False


# Helper functions for document processing
def extract_text_from_pdf(file_content):
    """Extract text from PDF file content."""
    text = ""
    with fitz.open(stream=file_content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_docx(file_content):
    """Extract text from DOCX file content."""
    if docx is None:
        st.warning("python-docx is not installed. DOCX processing is disabled.")
        return "DOCX processing is disabled. Please install python-docx library."

    doc = docx.Document(io.BytesIO(file_content))
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text


def extract_text_from_csv(file_content):
    """Extract text from CSV file content."""
    text = ""
    csv_data = pd.read_csv(io.BytesIO(file_content))
    # Convert the DataFrame to a string representation
    text = csv_data.to_string(index=False)
    return text


def extract_text_from_txt(file_content):
    """Extract text from TXT file content."""
    return file_content.decode('utf-8')


def process_file(uploaded_file):
    """Process an uploaded file and extract its text content."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_content = uploaded_file.getvalue()

    try:
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        elif file_extension == 'docx':
            text = extract_text_from_docx(file_content)
        elif file_extension == 'csv':
            text = extract_text_from_csv(file_content)
        elif file_extension == 'txt':
            text = extract_text_from_txt(file_content)
        else:
            # Try to read as text for other file types
            text = file_content.decode('utf-8')

        return Document(page_content=text, metadata={"source": uploaded_file.name})
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return None


# Sidebar for API key and file uploads
with st.sidebar:
    st.header("Setup")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    # File uploader
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, CSV, TXT)",
        accept_multiple_files=True,
        type=["pdf", "docx", "csv", "txt"]
    )

    # Process files button
    if st.button("Process Files"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key!")
        elif not uploaded_files:
            st.error("Please upload at least one file!")
        else:
            with st.spinner("Processing files..."):
                try:
                    # Process each uploaded file
                    processed_documents = []
                    for uploaded_file in uploaded_files:
                        doc = process_file(uploaded_file)
                        if doc:
                            processed_documents.append(doc)

                    if processed_documents:
                        st.session_state.documents = processed_documents
                        st.session_state.file_processor_ready = True
                        st.success(f"Successfully processed {len(processed_documents)} documents!")
                    else:
                        st.error("No documents were successfully processed.")
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")

    # Initialize RAG system button
    if st.button("Initialize RAG System"):
        if not openai_api_key:
            st.error("Please enter your OpenAI API key!")
        elif not st.session_state.file_processor_ready:
            st.error("Please process files first!")
        else:
            with st.spinner("Initializing RAG system..."):
                try:
                    # Set the API key
                    os.environ["OPENAI_API_KEY"] = openai_api_key

                    # Initialize components
                    llm = ChatOpenAI(model="gpt-4o-mini")
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                    vector_store = InMemoryVectorStore(embeddings)

                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100,
                        add_start_index=True,
                    )
                    splits = text_splitter.split_documents(st.session_state.documents)

                    # Index documents
                    vector_store.add_documents(documents=splits)

                    # Create a custom RAG prompt template instead of using hub
                    system_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:
                    """

                    human_template = "{question}"

                    chat_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_template),
                        ("human", human_template)
                    ])


                    # Define application state
                    class State(TypedDict):
                        question: str
                        context: List[Document]
                        answer: str


                    # Define application steps
                    def retrieve(state: State):
                        retrieved_docs = vector_store.similarity_search(state["question"], k=4)
                        return {"context": retrieved_docs}


                    def generate(state: State):
                        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
                        # Create messages directly
                        messages = [
                            {"type": "system",
                             "content": f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext:\n{docs_content}\n\nQuestion:\n{state['question']}\n\nAnswer:"},
                            {"type": "human", "content": state["question"]}
                        ]
                        response = llm.invoke(messages)
                        return {"answer": response.content}


                    # Build the graph
                    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
                    graph_builder.add_edge(START, "retrieve")
                    graph = graph_builder.compile()

                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.graph = graph

                    st.success("RAG system initialized successfully! You can now ask questions.")

                except Exception as e:
                    st.error(f"Error initializing RAG system: {str(e)}")

    st.markdown("### About")
    st.markdown(
        "This application uses Retrieval Augmented Generation (RAG) "
        "to answer questions about your uploaded documents. "
        "It retrieves relevant information from the documents and "
        "generates accurate answers based on the content."
    )

# Display uploaded documents section
if st.session_state.documents:
    with st.expander("Uploaded Documents"):
        for i, doc in enumerate(st.session_state.documents):
            st.write(f"{i + 1}. {doc.metadata.get('source', 'Unknown document')}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i + 1}:**")
                    st.markdown(source)

# Chat input and processing
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if system is initialized
    if st.session_state.graph is None:
        with st.chat_message("assistant"):
            system_message = "⚠️ Please upload documents and initialize the RAG system first using the buttons in the sidebar."
            st.markdown(system_message)
            st.session_state.messages.append({"role": "assistant", "content": system_message})
    else:
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Invoke the RAG pipeline
                result = st.session_state.graph.invoke({"question": prompt})
                answer = result["answer"]
                context = result["context"]

                # Display the answer
                st.markdown(answer)

                # Prepare sources for expander
                sources = []
                for i, doc in enumerate(context):
                    source_doc = doc.metadata.get('source', 'Unknown source')
                    start_index = doc.metadata.get('start_index', 'N/A')
                    source_text = f"*From document: {source_doc} (index: {start_index})*\n\n{doc.page_content}"
                    sources.append(source_text)

                # Show sources in expander
                if sources:
                    with st.expander("View Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i + 1}:**")
                            st.markdown(source)

                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

# Instructions at the bottom
with st.expander("How to use this app"):
    st.markdown("""
    1. Enter your OpenAI API key in the sidebar
    2. Upload your documents (PDF, DOCX, CSV, TXT files)
    3. Click the "Process Files" button
    4. Click the "Initialize RAG System" button
    5. Ask questions about your documents in the chat input
    6. View AI-generated answers and their sources

    **Example questions you can ask:**
    - What are the main topics covered in these documents?
    - Can you summarize the key points about [specific topic]?
    - What does the document say about [specific term or concept]?
    - How are [topic A] and [topic B] related according to the documents?
    - What evidence is provided for [specific claim]?
    """)
