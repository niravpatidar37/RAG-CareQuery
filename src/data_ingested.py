### Document Structure - DATA Ingestion
import os
## Directory Loader - To read all files from a directory
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
## load all the text fies from the directory

def data_ingestion_from_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )

    bucket_name = os.getenv("S3_BUCKET_NAME")
    prefix = "carequery/"

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    local_files = os.getenv("S3_DOWNLOADED_FILES")

    os.makedirs(local_files,exist_ok=True )


    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".csv"):
            filename = os.path.basename(key)
            local_csv_path = os.path.join(local_files,filename)
            if not os.path.exists(local_csv_path):
                s3_obj = s3.get_object(Bucket=bucket_name, Key=key)
                csv_content = s3_obj["Body"].read().decode("utf-8")
                # Write to a temporary file
                with open(local_csv_path,'w', encoding="utf-8") as f:
                    f.write(csv_content)
    return local_files

def data_ingestion():
    csv_files = data_ingestion_from_s3()

    dir_loader = DirectoryLoader(
        csv_files,
        glob = "**/*.csv", # Pattern to match files
        loader_cls= CSVLoader, # loader class to use
        loader_kwargs={'encoding':'utf-8'},
        show_progress=False
    )

    documents =dir_loader.load()
    return documents


def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators=['\n\n',"\n"," ",""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    #Example of chunks
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content:{split_docs[0].page_content[:300]}...")
        print(f"Metadata:{split_docs[0].metadata}")
    return split_docs

