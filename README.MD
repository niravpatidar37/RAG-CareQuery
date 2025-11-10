🩺 CareQuery — AI-Powered Healthcare Assistant (RAG + Gemini + AWS S3)

CareQuery is an end-to-end Generative AI project that helps users query healthcare-related data using a Retrieval-Augmented Generation (RAG) pipeline.
It combines AWS S3 for data storage, LangChain for document handling, ChromaDB for vector storage, and Google Gemini (via API) for intelligent response generation.

🚀 Key Features

Automated Data Ingestion from AWS S3
Loads structured health data (e.g., symptom–disease mappings) directly from an S3 bucket and processes it dynamically.

Embeddings + Vector Store
Generates embeddings using Google Gemini and stores them efficiently in ChromaDB for semantic retrieval.

End-to-End RAG Pipeline
Supports document ingestion → embedding → storage → retrieval → answer generation.

Dynamic File Management
Automatically checks for existing local files (s3_data/) and only downloads new/updated ones to reduce redundancy.

Portable and Cloud-Native
Works seamlessly across environments — local, Hugging Face Spaces, or cloud servers — without modification.

Interactive Interface (Streamlit)
Query health data interactively, view retrieved context, and get AI-generated responses in real-time.

🧠 Project Architecture
            ┌──────────────────────────────┐
            │        AWS S3 (Data)         │
            │  (CSV symptom & condition)   │
            └────────────┬─────────────────┘
                         │
                         ▼
           ┌──────────────────────────────┐
           │  Data Ingestion Layer         │
           │  (boto3 + LangChain CSVLoader)│
           └────────────┬─────────────────┘
                         │
                         ▼
           ┌──────────────────────────────┐
           │   Embedding Layer             │
           │  (Google Gemini Embeddings)   │
           └────────────┬─────────────────┘
                         │
                         ▼
           ┌──────────────────────────────┐
           │   Vector Store (ChromaDB)    │
           │ Persistent Storage: data/     │
           └────────────┬─────────────────┘
                         │
                         ▼
           ┌──────────────────────────────┐
           │   Retrieval + Generation      │
           │  (LangChain + Gemini LLM)     │
           └──────────────────────────────┘

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/anujpatel-ai/CareQuery.git
cd CareQuery

2️⃣ Create Virtual Environment
python -m venv rag_pipeline
source rag_pipeline/bin/activate  # (Linux/Mac)
rag_pipeline\Scripts\activate     # (Windows)

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Add Environment Variables

Create a .env file in the project root:

GOOGLE_API_KEY = "your_google_api_key"
AWS_ACCESS_KEY_ID = "your_access_key"
AWS_SECRET_ACCESS_KEY = "your_secret_key"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "carequery"
S3_DOWNLOADED_FILES = "s3_data/"

🧩 Project Structure
📦 CareQuery
├── src/
│   ├── data_ingested.py       # Data ingestion from AWS S3
│   ├── embedding_manager.py   # Embedding generation logic
│   ├── vectorstore.py         # ChromaDB-based vector store
│   ├── retriever.py           # Context retrieval functions
│   ├── llm_pipeline.py        # Gemini-powered response generation
│   ├── main.py                # Main pipeline orchestration
│   └── app.py                 # Streamlit app entry point
│
├── data/                      # Local persisted vector store
├── s3_data/                   # Downloaded files from S3
├── requirements.txt
├── README.md
└── healthcare.ipynb           # Notebook for testing pipeline

🧪 How It Works

Data Ingestion
The system fetches all .csv files from s3://carequery/ using boto3.
Files are temporarily saved to s3_data/ and loaded via LangChain CSVLoader.

Embedding Generation
Each record is converted to a vector embedding using Google Gemini embeddings.

Vector Storage
Embeddings and documents are stored in a persistent ChromaDB collection (data/).

Query Execution
On user query, the relevant context is retrieved from ChromaDB and passed to Gemini for response generation.

💻 Run the Streamlit App
streamlit run src/app.py