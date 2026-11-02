class SynthesisAgent:
    def __init__(self, llm):
        self.llm = llm

    def synthesize_response(self, context, query, intent):
        if intent == "CHITCHAT":
            sys_prompt = "You are a helpful assistant. Keep your response brief."
            return self.llm.generate(sys_prompt, query, max_tokens=150, temperature=0.6)
            
        base_rules = "Use ONLY the Context provided. If the answer is not there, say 'I cannot answer this based on the documents.'"
        
        intent_prompts = {
            "PROCEDURE": "Provide step-by-step instructions using numbered lists.",
            "COMPARE": "Compare the entities using a Markdown table.",
            "SUMMARIZE": "Provide a high-level summary using bullet points.",
            "DOC": "Answer directly and concisely."
        }
        
        specific_rule = intent_prompts.get(intent, intent_prompts["DOC"])
        sys_prompt = f"You are a strict technical assistant. {base_rules} {specific_rule}"
        user_prompt = f"Context:\n{context}\n\nQuery: {query}"
        
        return self.llm.generate(sys_prompt, user_prompt, max_tokens=300)