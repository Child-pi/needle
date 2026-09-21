# 🌲 Needle 3 PyTorch Implementation & Experiment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Child-pi/needle/blob/pytorch_experiment/pytorch_experiment.ipynb)

本分支 (`pytorch_experiment`) 為 Needle 3 的 **PyTorch 完整移植與架構實驗版本**。

原生 Needle 3 模型在預訓練階段使用 Google JAX/Flax，而在邊緣端部署時使用 `.cact` 二進制權重配合純 C 引擎執行。本分支將其神經網路架構完整移植至 **PyTorch (`torch.nn.Module`)**，方便機器學習研究人員在 PyTorch 生態系統下進行前向推論、自回歸生成、反向傳播微調與架構探索。

---

## 🌟 移植的核心架構元件

1. **`ZCRMSNorm`** (`torch.nn.Module`):
   - Zero-Centered RMSNorm，以零初始化尺度參數（Scale），確保在初始化時乘數剛好為 $1.0$。
2. **`HadamardMLP`** (`torch.nn.Module`):
   - Monarch Hadamard MLP，取代傳統厚重的 FFN，採用 Walsh-Hadamard 旋轉因子矩陣、Kronecker 積運算 (`_kron_apply`) 與條件調製向量 (`cond_v, cond_u`)。
3. **`MultiHeadAttention`** (`torch.nn.Module`):
   - 支援 GQA (Grouped-Query Attention)、因果深度卷積抽頭 (`qkv_conv_taps`)、RoPE (Rotary Position Embeddings) 與門控輸出投影 (`gate_proj`)。
4. **`NeedleModel` & `NeedleForCausalLM`** (`torch.nn.Module`):
   - 支援 2 ~ 20 層任意階梯式深度切片（Laddered Rungs）。
   - 內建校準置信度預測頭 (`confidence_proj`) 與自回歸生成器 (`generate()`)。

---

## 🚀 快速上手 (Quick Start)

### 1. 運行 Python 示範腳本

```bash
python needle/pytorch/run_example.py
```

輸出範例：
```text
======================================================================
🌲 Needle 3 PyTorch Architecture Demo
======================================================================
📦 Model Configuration:
   - Layers: 4 (Ladder rung)
   - Hidden Dim (d_model): 768
   - Attention Heads: 12 (Query) / 2 (KV, GQA)
   - Vocabulary: 16384
⚡ Running on device: mps / cuda
📊 Total Model Parameters: 19.88 Million

🚀 Executing Forward Pass...
   ✓ Output Logits Shape: [2, 16, 16384]  [Batch, SeqLen, VocabSize]
   ✓ Calibrated Confidence Scores: [0.472, 0.551]

🔤 Autoregressive Token Generation:
   - Generated Length: 16 tokens

🏋️ Training Step (Backpropagation & Optimizer):
   ✓ Cross-Entropy Loss: 755.5261
   ✓ Gradient Norm: 33.2816
   ✓ Optimizer Step Executed Successfully!
```

---

## 📓 Google Colab 雲端實驗與互動沙盒筆記本

我們為您提供了兩份互動式筆記本：

### 1. 🌲 官方同款互動沙盒 (Playground Demo - 類似 cactuscompute.com/needle)
內嵌 Web UI 互動介面，可點擊預設情境（智慧家居、音樂、手錶）、輸入指令並即時檢視 JSON 輸出、置信度與底層張量：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Child-pi/needle/blob/pytorch_experiment/needle_pytorch_playground.ipynb)

### 2. 🔬 PyTorch 模型架構深度實作 (Architecture Lab)
專注於 PyTorch 代碼層級的前向推論、自回歸生成、反向傳播梯度與 AdamW 優化器驗證：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Child-pi/needle/blob/pytorch_experiment/pytorch_experiment.ipynb)

### 3. ⚡ TensorFlow Lite for Microcontrollers (TFLM) 邊緣推論沙盒
展示 PyTorch 轉 TFLite、INT8 全整數量化、導出 C 標頭檔 (`xxd`) 與 TFLM 靜態記憶體池 (Tensor Arena) 邊緣遙測沙盒：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Child-pi/needle/blob/pytorch_experiment/needle_tflm_playground.ipynb)


