import json
import urllib.request
import pytest
import gguf
from pathlib import Path

def test_gguf_tokenizer_metadata_clean():
    """Verify GGUF file has proper token types and no raw unk byte tokens."""
    gguf_path = Path("models/mark-slm-v3-150m/mark_slm_v3_150m_iq4_nl.gguf")
    assert gguf_path.exists(), f"GGUF binary not found at {gguf_path}"

    reader = gguf.GGUFReader(str(gguf_path))
    assert "tokenizer.ggml.tokens" in reader.fields
    assert "tokenizer.ggml.token_type" in reader.fields
    assert "output.weight" in [t.name for t in reader.tensors]

    field_tt = reader.fields["tokenizer.ggml.token_type"]
    token_13_type = int(field_tt.parts[field_tt.data[13]][0])
    # Token 13 is newline (<0x0A>), must be typed as BYTE (6)
    assert token_13_type == int(gguf.TokenType.BYTE)

    # Bos (1), Eos (2), Pad (0)
    assert reader.fields["tokenizer.ggml.bos_token_id"].parts[-1][0] == 1
    assert reader.fields["tokenizer.ggml.eos_token_id"].parts[-1][0] == 2
    assert reader.fields["tokenizer.ggml.padding_token_id"].parts[-1][0] == 0

    del reader
    import gc
    gc.collect()


def test_generation_clean_output():
    """Verify inference generation contains zero raw byte fallback strings."""
    prompt = "Explain in one sentence what a computer is."
    req_data = {
        "model": "test-mark",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 30,
            "temperature": 0.6,
        }
    }

    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            output_text = data.get("response", "")
    except Exception as e:
        pytest.skip(f"Ollama server not reachable: {e}")

    # Mandatory assertions:
    assert "[UNK_BYTE_" not in output_text, f"Raw UNK_BYTE found in output: {output_text}"
    assert "<0x0A>" not in output_text, f"Raw <0x0A> found in output: {output_text}"
