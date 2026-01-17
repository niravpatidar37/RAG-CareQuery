import os
import boto3
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def data_ingestion_from_s3():
    """Downloads CSV files from S3 to local directory"""
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    bucket_name = os.getenv("S3_BUCKET_NAME")
    prefix = "carequery/" # Adjust prefix if needed, or empty string

    # Handle case where bucket might be empty or connection fails
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    except Exception as e:
        print(f"Error listing S3 objects: {e}")
        return os.getenv("S3_DOWNLOADED_FILES", "s3_data/")

    local_files = os.getenv("S3_DOWNLOADED_FILES", "s3_data/")
    os.makedirs(local_files, exist_ok=True)

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".csv"):
            filename = os.path.basename(key)
            local_csv_path = os.path.join(local_files, filename)
            
            # Simple caching: Don't redownload if exists (logic can be improved)
            if not os.path.exists(local_csv_path):
                print(f"Downloading {key}...")
                s3_obj = s3.get_object(Bucket=bucket_name, Key=key)
                csv_content = s3_obj["Body"].read().decode("utf-8")
                with open(local_csv_path, 'w', encoding="utf-8") as f:
                    f.write(csv_content)
            else:
                print(f"File {filename} already exists, skipping download.")
                
    return local_files

def data_ingestion():
    """Loads documents from local directory"""
    csv_files_dir = data_ingestion_from_s3()

    dir_loader = DirectoryLoader(
        csv_files_dir,
        glob="**/*.csv",
        loader_cls=CSVLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )

    documents = dir_loader.load()
    print(f"Loaded {len(documents)} documents from {csv_files_dir}")
    return documents

def split_documents(documents):
    """
    Split documents using Semantic Chunking.
    This groups sentences that are semantically similar.
    """
    print("Initializing Semantic Chunker...")
    
    # We need embeddings for Semantic Chunking to work
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Percentile assumes that sentences with similarity below x percentile are split points
    text_splitter = SemanticChunker(
        embeddings_model, 
        breakpoint_threshold_type="percentile" 
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} source docs into {len(split_docs)} semantic chunks")

    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:300]}...")
        print(f"Metadata: {split_docs[0].metadata}")
        
    return split_docs
