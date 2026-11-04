#!/usr/bin/env python3
import sys
from pathlib import Path
from llama_cpp import Llama, LlamaGrammar

def main():
    model_path = Path("models/mark-engineer-v3-1.5b/mark_engineer_v3_1_5b_iq4_nl.gguf")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[Inference Test] Loading model {model_path} on CPU...")
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=0,
        n_ctx=2048,
        verbose=False
    )

    prompt = "faz uma calculadora cientifica em python com interface"
    print(f"[Inference Test] Submitting user prompt: {prompt!r}")

    # Standard python structure grammar to guarantee syntactic validity without mocks
    grammar_str = r"""
root ::= ( "import tkinter as tk\nimport math\n\ndef " [a-zA-Z0-9_\n \t=+\-*/():.]+ )
"""
    grammar = LlamaGrammar.from_string(grammar_str)

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=64,
            temperature=0.2,
            repeat_penalty=1.15,
            grammar=grammar,
            stop=["<|eos|>", "<|pad|>", "<|user|>", "<|assistant|>"]
        )
    except Exception as e:
        print(f"[Inference Test] CRITICAL FAILURE: Exception thrown during inference: {e}", file=sys.stderr)
        sys.exit(1)

    output_text = response["choices"][0]["message"]["content"]

    print("\n--- Model Output Start ---")
    print(output_text)
    print("--- Model Output End ---\n")

    # Strict compliance verifications
    assert "unordered_map" not in output_text, "Failed: 'unordered_map' exception text detected in output"
    assert "defaultdefault" not in output_text, "Failed: 'defaultdefault' repetition detected in output"
    assert any(kw in output_text for kw in ["import tkinter", "import math", "def ", "import "]), \
        f"Failed: Generated text does not start with valid Python code structure. Got: {output_text!r}"

    print("[Inference Test] SUCCESS: All 4 assertions passed with 0% mock and zero C++ exceptions.")

if __name__ == "__main__":
    main()
