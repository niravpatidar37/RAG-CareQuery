### Simple RAG pipeline with groq LLM
from data_ingested import data_ingestion,split_documents
from vectors import VectorStore
from embedding import EmbeddingManager



def rag(query,retriever,llm,top_k=5,min_score=0.2,return_context=False):
    """
        RAG pipeline with extra features:
        -   Returns answer, sources, confidence score, and optionally full context.
    """
    results = retriever.retrieve(query,top_k=top_k,score_threshold= min_score)
    if not results:
        return {'answer': 'No relevant context found.','sources':[],'confidence':0.0,'context':''}

    # Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'sources': doc['metadata'].get('source_file',doc['metadata'].get('source','unknown')),
        'page': doc['metadata'].get('page','unknow'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]

    confidence = max([doc['similarity_score'] for doc in results])

    # generater answer
    prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion:{query}\n\nAnswer:"""
    response = llm.invoke([prompt.format(context=context,query=query)])

    output = {
        'answer': response.content,
        'sources':sources,
        'confidence':confidence
    }
    if return_context:
        output['context'] = context
    return output


def run_ingestion():
    print("Starting data ingestion...")

    # 1️⃣ Load all documents from folder
    documents = data_ingestion()

    # 2️⃣ Split documents into chunks for better RAG context
    chunks = split_documents(documents)

    # 3️⃣ Convert chunks to embeddings
    embedding_manager = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embedding(texts)

    # 4️⃣ Store in vector database
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)

    print("✅ Data successfully embedded and stored in Vector DB!")
    return vectorstore,embedding_manager

