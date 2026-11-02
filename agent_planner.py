class PlanningAgent:
    def __init__(self, llm):
        self.llm = llm

    def decompose_query(self, standalone_query, intent):
        if intent != "COMPARE":
            return [standalone_query]
            
        sys_prompt = "Break this comparison query into two separate search queries. Separate them with a pipe symbol (|). Output nothing else."
        user_prompt = f"Query: {standalone_query}\nSub-queries:"
        
        raw_output = self.llm.generate(sys_prompt, user_prompt, max_tokens=40)
        queries = [q.strip() for q in raw_output.split('|') if q.strip()]
        return queries if queries else [standalone_query]