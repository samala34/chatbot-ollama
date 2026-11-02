class RoutingAgent:
    def __init__(self, llm):
        self.llm = llm

    def route_query(self, standalone_query):
        sys_prompt = "Classify the query into one category: PROCEDURE, COMPARE, SUMMARIZE, CHITCHAT, or DOC. Output exactly ONE WORD."
        user_prompt = f"Query: {standalone_query}\nCategory:"
        
        intent = self.llm.generate(sys_prompt, user_prompt, max_tokens=5).upper()
        
        valid_intents = ["PROCEDURE", "COMPARE", "SUMMARIZE", "CHITCHAT", "DOC"]
        for valid in valid_intents:
            if valid in intent:
                return valid
        return "DOC" # Fallback