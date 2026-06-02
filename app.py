import streamlit as st
import os
import tempfile
from utils.document_loader import load_document
from utils.vector_store import VectorStoreManager
from utils.rag_chain import SimpleRAGChain

def main():
    st.set_page_config(page_title="RAG智能问答系统", page_icon="📚", layout="wide")
    
    st.title("📚 基于本地知识库的RAG智能问答系统")
    
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = VectorStoreManager()
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 文档上传")
        uploaded_files = st.file_uploader(
            "选择PDF或DOCX文件",
            type=["pdf", "docx"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"已选择 {len(uploaded_files)} 个文件")
            
        if st.button("🔄 构建知识库"):
            if not st.session_state.uploaded_files:
                st.warning("请先上传文档")
            else:
                with st.spinner("正在处理文档..."):
                    documents = []
                    for uploaded_file in st.session_state.uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        content = load_document(tmp_file_path)
                        os.unlink(tmp_file_path)
                        
                        if content.strip():
                            documents.append({
                                "filename": uploaded_file.name,
                                "content": content
                            })
                    
                    if documents:
                        chunk_count = st.session_state.vector_store_manager.add_documents(documents)
                        st.success(f"✅ 知识库构建完成！共添加 {chunk_count} 个文本块")
                        st.session_state.rag_chain = SimpleRAGChain(st.session_state.vector_store_manager)
                    else:
                        st.error("未能解析任何文档内容")
        
        st.subheader("📊 知识库状态")
        chunk_count = st.session_state.vector_store_manager.get_document_count()
        st.metric(label="文本块数量", value=chunk_count)
        
        if chunk_count > 0:
            if st.button("🗑️ 清空知识库"):
                st.session_state.vector_store_manager.clear_collection()
                st.session_state.rag_chain = None
                st.session_state.chat_history = []
                st.success("知识库已清空")
                st.rerun()
    
    with col2:
        st.subheader("💬 问答交互")
        
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
        
        user_question = st.text_input("请输入您的问题：", key="question_input")
        
        if st.button("提问") or user_question:
            if not user_question.strip():
                st.warning("请输入问题")
            elif not st.session_state.rag_chain:
                st.warning("请先上传文档并构建知识库")
            else:
                with st.spinner("正在思考..."):
                    answer = st.session_state.rag_chain.ask(user_question)
                    
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })
                    
                    st.rerun()
        
        if st.session_state.chat_history:
            if st.button("🧹 清空对话历史"):
                st.session_state.chat_history = []
                st.rerun()

if __name__ == "__main__":
    main()