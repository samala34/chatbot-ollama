# app.py
from config import OLLAMA_MODEL, DATA_DIR, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, EMBEDDING_MODEL_PATH, TOP_K_RETRIEVE, TOP_K_PARENTS, RRF_K
from document_processor import load_and_process_documents
from retriever import HierarchicalHybridRetriever

# Import our multi-agent system
from base_llm import BaseLLM
from agent_context import ContextAgent
from agent_router import RoutingAgent
from agent_planner import PlanningAgent
from agent_synthesizer import SynthesisAgent
from agent_verifier import VerificationAgent

def format_history(history_list):
    if not history_list: return ""
    return "\n".join([f"User: {u}\nBot: {b}" for u, b in history_list])

def main():
    print("1/3: Connecting to Ollama Base Engine...")
    # Pass the Ollama model name instead of a local folder path
    llm = BaseLLM(OLLAMA_MODEL) 
    
    # Initialize Micro-Agents using the shared LLM core
    context_agent = ContextAgent(llm)
    routing_agent = RoutingAgent(llm)
    planning_agent = PlanningAgent(llm)
    synthesis_agent = SynthesisAgent(llm)
    verification_agent = VerificationAgent(llm)
    
    print("2/3: Loading Hierarchical Hybrid Retriever...")
    retriever = HierarchicalHybridRetriever(EMBEDDING_MODEL_PATH)
    
    print("3/3: Processing Documents...")
    parent_store, child_chunks = load_and_process_documents(
        DATA_DIR, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
    )
    
    if child_chunks:
        print(f"Building vector and BM25 indices...")
        retriever.ingest(parent_store, child_chunks)
    else:
        print(f"\n[Warning] No documents found in '{DATA_DIR}'. RAG disabled.")
    
    print("\n" + "="*50)
    print(f" Multi-Agent RAG Ready (Powered by Ollama {OLLAMA_MODEL}) ")
    print("="*50)
    
    chat_history = []
    
    while True:
        raw_query = input("\nYou: ")
        if raw_query.lower() in ['exit', 'quit']: break
            
        print("\n⚙️ Orchestrating...")
        
        # 1. Context (Memory)
        history_str = format_history(chat_history)
        standalone_query = context_agent.rewrite_query(raw_query, history_str)
        if standalone_query != raw_query:
            print(f"   [Context Agent] -> '{standalone_query}'")
            
        # 2. Router
        intent = routing_agent.route_query(standalone_query)
        print(f"   [Routing Agent] -> {intent}")
        
        # CHITCHAT Fast-Path
        if intent == "CHITCHAT":
            draft = synthesis_agent.synthesize_response("", standalone_query, intent)
            print(f"\nBot: {draft}")
            chat_history.append((raw_query, draft))
            if len(chat_history) > 3: chat_history.pop(0)
            continue
            
        if not child_chunks:
            print("Bot: I need documents to answer technical queries.")
            continue
            
        # 3. Planner
        search_queries = planning_agent.decompose_query(standalone_query, intent)
        if len(search_queries) > 1:
            print(f"   [Planning Agent] -> {search_queries}")
            
        # 4. Search (Hybrid Retrieval)
        retrieved_contexts = []
        for sq in search_queries:
            chunks = retriever.retrieve(sq, top_k_retrieve=TOP_K_RETRIEVE, top_k_parents=TOP_K_PARENTS, rrf_k=RRF_K)
            retrieved_contexts.extend(chunks)
            
        unique_contexts = list(set(retrieved_contexts))
        context_block = "\n---\n".join(unique_contexts)
        print(f"   [Search Engine] -> Retrieved {len(unique_contexts)} unique parent chunks.")
        
        # 5. Synthesis
        print("   [Synthesis Agent] -> Drafting response...")
        draft = synthesis_agent.synthesize_response(context_block, standalone_query, intent)
        
        # 6. Verification
        print("   [Verification Agent] -> Checking for hallucinations...")
        final_response = verification_agent.verify_response(draft, context_block)
        
        # Append Citations
        if "cannot answer this" not in final_response.lower() and "withheld the response" not in final_response.lower():
            sources = list(set([ctx.split('\n')[1] for ctx in unique_contexts if '\n' in ctx]))
            if sources:
                final_response += "\n\n**Sources:**\n- " + "\n- ".join(sources)
        
        print(f"\nBot:\n{final_response}")
        
        chat_history.append((raw_query, final_response))
        if len(chat_history) > 3: chat_history.pop(0)

if __name__ == "__main__":
    main()