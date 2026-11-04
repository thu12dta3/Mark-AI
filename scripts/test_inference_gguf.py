#!/usr/bin/env python3
import json
import urllib.request
import sys

def test_inference():
    prompt = "Explain in one sentence what a computer is."
    print(f"Testing inference on Mark-SLM-V3-150M GGUF with prompt: {prompt!r}")

    req_data = {
        "model": "test-mark",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 40,
            "temperature": 0.6,
        }
    }

    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(req_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        output_text = data.get("response", "")

    print("\n--- Model Output Start ---")
    print(repr(output_text))
    print("--- Model Output End ---\n")

    assert "[UNK_BYTE_" not in output_text, f"Failure: found raw byte token in output: {output_text}"
    assert "<0x0A>" not in output_text, f"Failure: found raw newline token in output: {output_text}"

    print("SUCCESS: Zero raw byte fallbacks detected. Output clean.")

if __name__ == "__main__":
    test_inference()
