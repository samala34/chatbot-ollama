class VerificationAgent:
    def __init__(self, llm):
        self.llm = llm

    def verify_response(self, draft, context):
        if "cannot answer this" in draft.lower() or not context:
            return draft
            
        sys_prompt = "Does the Draft directly contradict the Context or invent facts not present in the Context? Answer strictly YES or NO."
        user_prompt = f"Context:\n{context}\n\nDraft:\n{draft}\n\nAnswer:"
        
        judgment = self.llm.generate(sys_prompt, user_prompt, max_tokens=5).upper()
        
        if "YES" in judgment:
            return "I generated an answer, but upon self-verification, I realized it contained information not verified by the documentation. I have withheld the response."
        return draft