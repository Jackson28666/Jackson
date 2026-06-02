from utils.document_loader import load_documents_from_folder
from utils.vector_store import VectorStoreManager
from utils.rag_chain import SimpleRAGChain

def test_rag_pipeline():
    print("=== 测试RAG问答系统 ===")
    
    vs_manager = VectorStoreManager()
    
    sample_docs = [
        {
            "filename": "nlp_intro.txt",
            "content": """自然语言处理（Natural Language Processing，简称NLP）是人工智能领域的一个重要分支，它致力于使计算机能够理解、解释和生成人类语言。

NLP的主要任务包括：
1. 文本分类：将文本分为不同的类别
2. 情感分析：识别文本中的情感倾向
3. 命名实体识别：识别文本中的实体如人名、地名、机构名等
4. 机器翻译：将一种语言翻译成另一种语言
5. 问答系统：回答用户提出的问题

近年来，随着深度学习的发展，NLP取得了显著进步，特别是大语言模型如GPT、BERT等的出现，极大地推动了NLP技术的发展。

词向量是NLP中的基础概念，它将词语表示为向量形式，使得计算机能够处理文本数据。常见的词向量模型包括Word2Vec、GloVe等。

注意力机制是Transformer架构的核心，它使得模型能够在处理序列数据时关注不同位置的信息。"""
        }
    ]
    
    print("\n1. 添加测试文档到向量库...")
    vs_manager.add_documents(sample_docs)
    print(f"向量库中文本块数量: {vs_manager.get_document_count()}")
    
    print("\n2. 测试检索功能...")
    results = vs_manager.search("什么是自然语言处理", k=3)
    print(f"检索到 {len(results)} 个相关文本块")
    
    print("\n3. 测试RAG问答...")
    rag_chain = SimpleRAGChain(vs_manager)
    
    test_questions = [
        "什么是自然语言处理？",
        "NLP有哪些主要任务？",
        "什么是词向量？",
        "注意力机制是什么？",
        "什么是深度学习？",
        "今天天气怎么样？"
    ]
    
    for question in test_questions:
        print(f"\n问题: {question}")
        answer = rag_chain.ask(question)
        print(f"回答: {answer}")

if __name__ == "__main__":
    test_rag_pipeline()