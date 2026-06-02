# RAG智能问答系统

基于本地知识库的RAG（检索增强生成）智能问答系统，使用Ollama本地大模型、LangChain框架和Streamlit构建。

## 项目功能

- 支持上传PDF和DOCX文档
- 自动进行文档解析、文本分块和向量化存储
- 基于Chroma向量数据库进行相似性检索
- 利用Ollama本地大模型进行问答
- 支持多轮对话，保持上下文记忆

## 环境要求

- Python 3.9+
- Ollama（已安装并运行）
- 推荐模型：deepseek-r1:7b 或 qwen2:7b

## 使用说明

```bash
pip install -r requirements.txt
streamlit run app.py
```