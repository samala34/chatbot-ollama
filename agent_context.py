class ContextAgent:
    def __init__(self, llm):
        self.llm = llm

    def rewrite_query(self, query, history):
        if not history:
            return query
            
        sys_prompt = "You are a query rewriting assistant. Given a chat history and a follow-up query, rewrite the query to be completely standalone. Replace pronouns with specific nouns. Output ONLY the rewritten query."
        user_prompt = f"History: {history}\nFollow-up: {query}\nRewritten:"
        
        return self.llm.generate(sys_prompt, user_prompt, max_tokens=30)