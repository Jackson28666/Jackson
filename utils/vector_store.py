import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

class VectorStoreManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection_name = "nlp_documents"
        
    def get_or_create_collection(self):
        try:
            collection = self.client.get_collection(self.collection_name)
        except (ValueError, chromadb.errors.NotFoundError):
            collection = self.client.create_collection(self.collection_name)
        return collection
    
    def add_documents(self, documents):
        collection = self.get_or_create_collection()
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        doc_id = 0
        for doc in documents:
            chunks = self.text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "filename": doc["filename"],
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
                all_ids.append(f"doc_{doc_id}_chunk_{i}")
            doc_id += 1
        
        if all_chunks:
            collection.add(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"Added {len(all_chunks)} chunks to vector store")
        return len(all_chunks)
    
    def search(self, query, k=3):
        collection = self.get_or_create_collection()
        if collection.count() == 0:
            return []
        
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        return [
            {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]
    
    def get_document_count(self):
        collection = self.get_or_create_collection()
        return collection.count()
    
    def clear_collection(self):
        self.client.delete_collection(self.collection_name)
        print("Collection cleared")