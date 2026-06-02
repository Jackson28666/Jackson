from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

class RAGChain:
    def __init__(self, vector_store_manager, model_name="deepseek-r1:7b"):
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        
        self.llm = Ollama(model=model_name)
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""基于以下参考文档回答问题：

{context}

请根据以上参考文档回答问题：{question}

如果文档中没有相关信息，请明确说"文档中未找到相关答案"，不要编造答案。"""
        )
        
        self.chain = None
    
    def build_chain(self):
        from langchain.chains import RetrievalQA
        from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
        
        retriever = self.vector_store_manager.client.as_retriever(
            collection_name=self.vector_store_manager.collection_name,
            search_kwargs={"k": 3}
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            return_source_documents=True
        )
    
    def ask(self, question):
        if not self.chain:
            self.build_chain()
        
        try:
            result = self.chain({"question": question})
            answer = result.get("answer", "")
            
            if not answer.strip() or "未找到" in answer or "不知道" in answer:
                return "文档中未找到相关答案"
            
            return answer
        except Exception as e:
            print(f"Error in RAG chain: {e}")
            return f"回答问题时发生错误: {str(e)}"
    
    def clear_history(self):
        self.memory.clear()

class SimpleRAGChain:
    def __init__(self, vector_store_manager, model_name="deepseek-r1:7b"):
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        self.llm = Ollama(model=model_name)
    
    def ask(self, question):
        context = self.vector_store_manager.search(question, k=3)
        
        if not context:
            return "文档中未找到相关答案"
        
        context_text = "\n\n".join([item["content"] for item in context])
        
        prompt = f"""基于以下参考文档回答问题：

{context_text}

请根据以上参考文档回答问题：{question}

如果文档中没有相关信息，请明确说"文档中未找到相关答案"，不要编造答案。"""
        
        try:
            response = self.llm(prompt)
            if not response.strip() or "未找到" in response or "不知道" in response or "无法回答" in response:
                return "文档中未找到相关答案"
            return response
        except Exception as e:
            print(f"Error in SimpleRAGChain: {e}")
            return f"回答问题时发生错误: {str(e)}"