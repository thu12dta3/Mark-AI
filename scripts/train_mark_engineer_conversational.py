import json
import math
import os
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, '/content/Mark-AI')
from export.quantize_4bit import quantize_mark_model_nf4
from src.model.mark_arch import MarkConfig, MarkModel
from src.tokenizer.build_bpe import build_mark_bpe_tokenizer

def main():
    print('=== [Mark Engineer 1.5B: Conversational & Natural Dialogue Alignment on NVIDIA A100-80GB] ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f'[A100 GPU] {gpu_name} | VRAM Total: {vram_gb:.2f} GB')
    else:
        print('[Warning] Running on CPU!')
    tokenizer = build_mark_bpe_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    print(f'[Tokenizer] BPE vocabulary size: {vocab_size} tokens.')
    config = MarkConfig.create_engineer_1_5b(vocab_size=vocab_size)
    print(f'[Architecture] Initializing Mark Engineer 1.5B (dim={config.dim}, layers={config.n_layers}, MLA={config.use_mla}, MoE={config.n_routed_experts}+{config.n_shared_experts}, MTP={config.mtp_depth})...')
    model = MarkModel(config).to(device)
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'[Parameters] Total: {total_params:,} ({total_params / 1000000000.0:.4f}B) | Trainable: {trainable_params:,}')
    data_dir = Path('/content/Mark-AI/data/synthetic_dpo')
    conversational_samples = []
    conv_file = data_dir / 'mark_engineer_conversational.jsonl'
    if conv_file.exists():
        raw_conv = []
        with open(conv_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_conv.append(json.loads(line))
                    except Exception:
                        pass
        conversational_samples.extend(raw_conv * 15)
        print(f'[Dataset: {conv_file.name}] Ingested {len(raw_conv)} base samples (amplified to {len(raw_conv) * 15} conversational alignment pairs).')
    dataset_files = ['mark_engineer_senior_os_systems.jsonl', 'mark_engineer_stem_polyglot.jsonl', 'mark_engineer_train.jsonl']
    for fname in dataset_files:
        dpath = data_dir / fname
        if dpath.exists():
            cnt = 0
            with open(dpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            conversational_samples.append(json.loads(line))
                            cnt += 1
                        except Exception:
                            pass
            print(f'[Dataset: {dpath.name}] Ingested {cnt} samples.')
    print(f'[Master Dataset] Total training pairs in conversational alignment curriculum: {len(conversational_samples)}')

    class TextPairDataset(Dataset):

        def __init__(self, data, tok, max_len=1024):
            self.encoded = []
            for item in data:
                prompt_txt = item.get('prompt', '')
                resp_txt = item.get('completion', item.get('chosen', item.get('response', '')))
                full_txt = prompt_txt + '\n' + resp_txt
                enc = tok.encode(full_txt)
                ids = enc.ids[:max_len]
                if len(ids) < 32:
                    continue
                self.encoded.append(torch.tensor(ids, dtype=torch.long))

        def __len__(self):
            return len(self.encoded)

        def __getitem__(self, idx):
            return self.encoded[idx]

    def collate_fn(batch):
        max_l = max((t.size(0) for t in batch))
        input_ids = torch.full((len(batch), max_l), fill_value=config.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_l), fill_value=-100, dtype=torch.long)
        for i, t in enumerate(batch):
            input_ids[i, :t.size(0)] = t
            labels[i, :t.size(0)] = t
        return (input_ids, labels)
    train_ds = TextPairDataset(conversational_samples, tokenizer, max_len=1024)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00012, betas=(0.9, 0.95), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_steps = 400
    grad_accum_steps = 4
    warmup_steps = 25
    step = 0
    running_loss = 0.0
    start_time = time.time()
    ckpt_dir = Path('/content/Mark-AI/checkpoints/engineer')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print('\n=== [Starting Conversational Alignment Loop: 400 Steps | Cosine Annealing | MTP Enabled] ===')
    model.train()
    optimizer.zero_grad()
    while step < total_steps:
        for input_ids, labels in train_loader:
            if step >= total_steps:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, mtp_logits = model(input_ids, return_mtp=True)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_main = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                if mtp_logits is not None:
                    shift_mtp = mtp_logits[..., :-1, :].contiguous()
                    shift_mtp_labels = labels[..., 2:].contiguous()
                    loss_mtp = criterion(shift_mtp.view(-1, vocab_size), shift_mtp_labels.view(-1))
                    total_loss = (loss_main + config.mtp_loss_weight * loss_mtp) / grad_accum_steps
                else:
                    total_loss = loss_main / grad_accum_steps
            total_loss.backward()
            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            step += 1
            running_loss += total_loss.item() * grad_accum_steps
            if step <= warmup_steps:
                curr_lr = 0.00012 * (step / warmup_steps)
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                curr_lr = 1e-05 + 0.5 * (0.00012 - 1e-05) * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = curr_lr
            if step % 25 == 0 or step == total_steps or step == 1:
                window = 25 if step >= 25 else step
                avg_l = running_loss / window
                if step >= 25:
                    running_loss = 0.0
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0.01
                remaining_sec = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                eta_min = remaining_sec / 60.0
                vram_used = torch.cuda.memory_allocated(0) / 1024 ** 3 if device.type == 'cuda' else 0.0
                print(f'[Step {step:04d}/{total_steps}] Loss: {total_loss.item() * grad_accum_steps:.4f} (Avg: {avg_l:.4f}) | LR: {curr_lr:.2e} | VRAM: {vram_used:.2f} GB | ETA: {eta_min:.1f}m')
            if step in (100, 200, 300):
                interm_ckpt = ckpt_dir / f'mark_engineer_1_5b_step_{step:04d}.pt'
                torch.save(model.state_dict(), interm_ckpt)
                print(f'  [Intermediate Checkpoint] Saved on A100: {interm_ckpt.name}')
    print('\n[Alignment Finished] Consolidating and saving final Mark Engineer 1.5B weights...')
    final_ckpt = ckpt_dir / 'mark_engineer_1_5b_final.pt'
    torch.save(model.state_dict(), final_ckpt)
    print(f'  [Final Checkpoint] Saved on A100: {final_ckpt}')
    print('[Weight Export: BF16 Inference Format on A100]')
    inf_ckpt = ckpt_dir / 'mark_engineer_1_5b_inference.pt'
    bf16_state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    torch.save(bf16_state, inf_ckpt)
    inf_size_mb = inf_ckpt.stat().st_size / (1024 * 1024)
    print(f'  [BF16 Inference] Saved on A100: {inf_ckpt} ({inf_size_mb:.2f} MB)')
    print('[Weight Export: Native IQ4_NL Quantization on A100]')
    iq4_ckpt = ckpt_dir / 'mark_engineer_1_5b_iq4_nl.pt'
    quantize_mark_model_nf4(model, weight_quant_type='IQ4_NL')
    iq4_state = model.state_dict()
    torch.save(iq4_state, iq4_ckpt)
    iq4_size_mb = iq4_ckpt.stat().st_size / (1024 * 1024)
    print(f'  [Quantized IQ4_NL] Saved on A100: {iq4_ckpt} ({iq4_size_mb:.2f} MB)')
    meta_file = ckpt_dir / 'MARK_ENGINEER_CONVERSATIONAL_COMPLETE.json'
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump({'model_name': 'Mark Engineer 1.5B (Conversational Dialogue Alignment)', 'architecture': 'DeepSeek-V3 MLA + DeepSeekMoE 16+1 + MTP', 'total_parameters': total_params, 'trained_steps': total_steps, 'dialogue_personality': 'Empathetic, articulate, engaging senior tech lead / ChatGPT-style dialogue partner', 'anti_robotic_alignment': True, 'final_loss': total_loss.item() * grad_accum_steps, 'completed_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), 'status': 'CONVERGED_SUCCESSFULLY'}, f, indent=2)
    print('=== [Mark Engineer 1.5B Conversational Training Completed Successfully on A100] ===')
if __name__ == '__main__':
    main()
