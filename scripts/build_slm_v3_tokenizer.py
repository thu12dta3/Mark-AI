import json
from pathlib import Path
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models" / "mark-slm-v3-150m"

def main():
    print("Loading base BPE tokenizer from mistralai/Mistral-7B-v0.1...")
    raw = json.loads(Tokenizer.from_pretrained("mistralai/Mistral-7B-v0.1").to_str())

    special_tokens_list = [
        "<|pad|>", "<|bos|>", "<|eos|>", "<think>", "</think>",
        "[RETRIEVE]", "[/RETRIEVE]", "[EVALUATE]", "[ACCEPT]", "[REJECT]",
        "<|vision_start|>", "<|vision_patch|>", "<|vision_end|>", "<|voice_stream|>",
        "[UNKNOWN]", "[WEB_SEARCH]", "[/WEB_SEARCH]",
        "<|rag_context|>", "<|/rag_context|>", "<|mcp_call|>", "<|mcp_response|>",
        "<|citation|>", "<|/citation|>",
        "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
        "<|arch_plan|>", "<|/arch_plan|>",
        "<|extra_0|>", "<|extra_1|>", "<|extra_2|>", "<|extra_3|>",
        "<|extra_4|>", "<|extra_5|>", "<|extra_6|>"
    ]

    new_vocab = {}
    for k, v in raw["model"]["vocab"].items():
        if v == 0:
            new_vocab["<|pad|>"] = 0
        elif v == 1:
            new_vocab["<|bos|>"] = 1
        elif v == 2:
            new_vocab["<|eos|>"] = 2
        else:
            new_vocab[k] = v

    curr_id = 32000
    for tok_name in special_tokens_list[3:]:
        new_vocab[tok_name] = curr_id
        curr_id += 1

    assert len(new_vocab) == 32032, f"Expected 32032, got {len(new_vocab)}"
    raw["model"]["vocab"] = new_vocab

    raw["added_tokens"] = [
        {"id": 0, "content": "<|pad|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 1, "content": "<|bos|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": 2, "content": "<|eos|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
    ]
    for i, tok_name in enumerate(special_tokens_list[3:], start=32000):
        raw["added_tokens"].append({
            "id": i, "content": tok_name, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True
        })

    raw["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<|bos|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "pair": [
            {"SpecialToken": {"id": "<|bos|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "<|bos|>", "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}
        ],
        "special_tokens": {"<|bos|>": {"id": "<|bos|>", "ids": [1], "tokens": ["<|bos|>"]}}
    }

    tok_file = MODEL_DIR / "tokenizer.json"
    with open(tok_file, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"Saved {tok_file} successfully ({tok_file.stat().st_size:,} bytes).")

    # Update tokenizer_config.json
    tok_cfg_file = MODEL_DIR / "tokenizer_config.json"
    with open(tok_cfg_file, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)

    tok_cfg["vocab_size"] = 32032
    tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    tok_cfg["bos_token"] = "<|bos|>"
    tok_cfg["eos_token"] = "<|eos|>"
    tok_cfg["pad_token"] = "<|pad|>"
    tok_cfg["pad_token_id"] = 0
    tok_cfg["bos_token_id"] = 1
    tok_cfg["eos_token_id"] = 2

    with open(tok_cfg_file, "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, ensure_ascii=False, indent=2)
    print(f"Updated {tok_cfg_file} successfully.")

if __name__ == "__main__":
    main()
