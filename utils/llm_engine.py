"""
Utility — LLM Engine (Ollama backend)

Replaces FLAN-T5 with Llama 3.2 running locally via Ollama.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull the model: ollama pull llama3.2
    3. Ollama runs automatically as a local server on port 11434

No API key needed. Completely free and local.
"""

import requests


class LLMEngine:
    def __init__(self, model: str = "llama3.2"):
        self.model   = model
        self.url     = "http://localhost:11434/api/generate"
        self._verify_connection()

    def _verify_connection(self):
        """Check Ollama is running and the model is available."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m["name"] for m in resp.json().get("models", [])]
            base_names = [m.split(":")[0] for m in models]
            if self.model not in base_names:
                print(f"⚠️  Model '{self.model}' not found in Ollama.")
                print(f"   Run: ollama pull {self.model}")
                print(f"   Available: {base_names}")
            else:
                print(f"LLM ready: {self.model} via Ollama ✅")
        except requests.exceptions.ConnectionError:
            print("❌ Ollama is not running.")
            print("   Start it with: ollama serve")
            print("   Or just open the Ollama desktop app.")
            raise SystemExit(1)

    def generate(self, prompt: str, max_length: int = 150) -> str:
        """
        Send a prompt to Ollama and return the response text.
        max_length maps to num_predict (approximate token count).
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_length,
                "temperature": 0.7,
                "top_p":       0.9,
                "top_k":       40,
                "stop":        ["\n\n", "---"],
            }
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["response"].strip()
        except requests.exceptions.Timeout:
            print("⚠️  Ollama request timed out. Returning fallback.")
            return "Search academic databases for relevant papers"
        except Exception as e:
            print(f"⚠️  Ollama error: {e}")
            return "Search academic databases for relevant papers"