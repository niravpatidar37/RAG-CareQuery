import os
import boto3
from langchain_community.document_loaders import DirectoryLoader, CSVLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def data_ingestion_from_s3():
    """Downloads CSV files from S3 to local directory"""
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION")
    bucket_name = os.getenv("S3_BUCKET_NAME")

    if not all([access_key, secret_key, region, bucket_name]):
        print("❌ Error: Missing AWS credentials or S3_BUCKET_NAME in environment.")
        return os.getenv("S3_DOWNLOADED_FILES", "s3_data/")

    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    ) 

    if not bucket_name:
        print("❌ Error: S3_BUCKET_NAME is not set in your .env file.")
        return os.getenv("S3_DOWNLOADED_FILES", "s3_data/")

    # Handle case where bucket might be empty or connection fails
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    except Exception as e:
        print(f"❌ Error listing S3 objects: {e}")
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
    """Loads documents from S3 cache or local data directory"""
    # 1. Attempt S3 sync (downlods to s3_data/)
    s3_files_dir = data_ingestion_from_s3()
    
    # 2. Define list of possible data sources
    search_paths = [s3_files_dir, "data/"]
    
    documents = []
    for path in search_paths:
        if not os.path.exists(path):
            continue
            
        print(f"Searching for data in: {path}...")
        
        # Try Loading with utf-8-sig first (handles BOM and most standard exports)
        try:
            dir_loader = DirectoryLoader(
                path,
                glob="**/*.csv",
                loader_cls=CSVLoader,
                loader_kwargs={'encoding': 'utf-8-sig'},
                show_progress=True
            )
            loaded_docs = dir_loader.load()
            print(f"Loaded {len(loaded_docs)} documents from {path}")
            documents.extend(loaded_docs)
        except Exception as e:
            if "charmap" in str(e) or "utf-8" in str(e):
                print(f"Notice: UTF-8 failed for {path}, trying latin1...")
                dir_loader = DirectoryLoader(
                    path,
                    glob="**/*.csv",
                    loader_cls=CSVLoader,
                    loader_kwargs={'encoding': 'latin1'},
                    show_progress=True
                )
                loaded_docs = dir_loader.load()
                print(f"Loaded {len(loaded_docs)} documents from {path} (latin1)")
                documents.extend(loaded_docs)
            else:
                print(f"Error loading from {path}: {e}")

    if not documents:
        print("⚠️ Warning: No CSV documents found in any of the search paths.")
        
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
