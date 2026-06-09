"""Test RAG pipeline end-to-end"""
import os
import sys

from utils.document_loader import load_documents_from_folder
from utils.vector_store import VectorStoreManager
from utils.rag_chain import SimpleRAGChain

def test_rag():
    print("Testing RAG pipeline...")
    
    vector_store = VectorStoreManager()
    
    documents = [{
        "filename": "test.txt",
        "content": "自然语言处理是人工智能的一个重要分支，涉及计算机与人类语言之间的交互。"
    }]
    
    chunk_count = vector_store.add_documents(documents)
    print(f"Added {chunk_count} chunks")
    
    rag_chain = SimpleRAGChain(vector_store)
    
    questions = [
        "什么是自然语言处理?",
        "人工智能有哪些分支?"
    ]
    
    for question in questions:
        answer = rag_chain.ask(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()

if __name__ == "__main__":
    test_rag()