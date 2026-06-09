class SimpleRAGChain:
    def __init__(self, vector_store_manager):
        self.vector_store_manager = vector_store_manager
    
    def ask(self, question):
        context = self.vector_store_manager.search(question, k=3)
        
        if not context:
            return "文档中未找到相关答案"
        
        context_text = "\n\n".join([item["content"] for item in context])
        
        keywords = [word for word in question.split() if len(word) > 1]
        found_content = []
        
        for item in context:
            content = item["content"]
            match_count = sum(1 for kw in keywords if kw.lower() in content.lower())
            if match_count > 0:
                found_content.append((match_count, content))
        
        if found_content:
            found_content.sort(key=lambda x: -x[0])
            best_match = found_content[0][1]
            return f"根据文档内容，找到相关信息：\n\n{best_match[:800]}..."
        else:
            return "文档中未找到相关答案"