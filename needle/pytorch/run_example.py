"""Needle 3 in PyTorch: Complete Execution and Training Example.

This script demonstrates how to:
1. Instantiate Needle 3 in PyTorch (customizable depth from 2 to 20 layers).
2. Perform forward inference (Logits + Calibrated Confidence).
3. Perform autoregressive token generation.
4. Perform gradient backpropagation with AdamW optimizer.
"""

import math
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn.functional as F
from needle.pytorch import NeedleConfig, NeedleForCausalLM


def main():
    print("=" * 70)
    print("🌲 Needle 3 PyTorch Architecture Demo")
    print("=" * 70)

    # 1. Initialize Configuration (e.g. 4-layer 29M or 8-layer 52M subnetwork)
    # Full flagship is 20 layers, d_model=768, num_heads=12
    config = NeedleConfig(
        vocab_size=16384,
        d_model=768,
        num_heads=12,
        num_kv_heads=2,
        num_layers=4,        # 4-layer fast subnetwork rung
        qk_head_dim=48,
        v_head_dim=64,
        max_seq_len=2048,
        qkv_conv_taps=3,     # Depthwise causal convolution on Q, K, V
        torch_dtype="float32",
    )

    print(f"📦 Model Configuration:")
    print(f"   - Layers: {config.num_layers} (Ladder rung)")
    print(f"   - Hidden Dim (d_model): {config.d_model}")
    print(f"   - Attention Heads: {config.num_heads} (Query) / {config.num_kv_heads} (KV, GQA)")
    print(f"   - Vocabulary: {config.vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Running on device: {device}")

    # 2. Build PyTorch Model
    model = NeedleForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total Model Parameters: {total_params / 1e6:.2f} Million")

    # 3. Forward Pass Demo
    batch_size = 2
    seq_len = 16
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    print("\n🚀 Executing Forward Pass...")
    outputs = model(dummy_input)
    logits = outputs["logits"]
    confidence = outputs["confidence"]

    print(f"   ✓ Output Logits Shape: {list(logits.shape)}  [Batch, SeqLen, VocabSize]")
    print(f"   ✓ Calibrated Confidence Scores: {confidence.detach().cpu().tolist()}")

    # 4. Autoregressive Generation Demo
    print("\n🔤 Autoregressive Token Generation:")
    prompt = torch.randint(0, config.vocab_size, (1, 6), device=device)
    print(f"   - Initial Prompt Tokens: {prompt[0].tolist()}")

    generated = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    print(f"   - Completed Sequence Tokens: {generated[0].tolist()}")
    print(f"   - Generated Length: {generated.shape[1]} tokens")

    # 5. Training Step with AdamW Demo
    print("\n🏋️ Training Step (Backpropagation & Optimizer):")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"   ✓ Cross-Entropy Loss: {loss.item():.4f}")
    print(f"   ✓ Gradient Norm: {grad_norm.item():.4f}")
    print(f"   ✓ Optimizer Step Executed Successfully!")

    print("\n" + "=" * 70)
    print("✅ PyTorch experiment completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
