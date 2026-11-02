# base_llm.py
import ollama

class BaseLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Connecting to local Ollama instance using model: '{self.model_name}'...")
        
        # Verify if Ollama is running and has the model
        try:
            ollama.show(self.model_name)
        except Exception as e:
            print(f"\n[Warning] Could not find '{self.model_name}' in Ollama.")
            print(f"Make sure Ollama is running and you have run: ollama pull {self.model_name}\n")

    def generate(self, system_prompt, user_prompt, max_tokens, temperature=0.1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Send chat request to Ollama
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens # Ollama's version of max_tokens
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            print(f"\n[Ollama Error] {e}")
            return "I encountered an error communicating with the Ollama service."