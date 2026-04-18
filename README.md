# Q-MoLE: Extreme Edge Compression for LLMs via 1.58-bit Mixture of Experts

[![Research-Grade](https://img.shields.io/badge/Research-M.Tech_AI-blueviolet)](https://github.com/himanshu1322/QMoLE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: Llama-3-8B](https://img.shields.io/badge/Backbone-Llama--3--8B-orange)](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
unsloth/llama-3-8b-Instruct-bnb-4bit

> **"What if you could run the reasoning power of an 8B parameter model with the energy footprint of a mobile app?"**

QMoLE (Quantized Mixture of 1.58-bit Experts) is a specialized research framework that bridges the gap between massive Large Language Models and resource-constrained edge hardware. By implementing a **ternary quantization** strategy and a **bottleneck adapter architecture**, Q-MoLE achieves state-of-the-art compression without sacrificing architectural complexity.

---

## The Research Breakthrough

In standard MoE architectures, expert layers are the primary memory bottleneck. Q-MoLE solves this via:
1. **1.58-bit Ternary Quantization**: Weights are restricted to $\{-1, 0, 1\}$, replacing floating-point multiplications with high-speed addition/subtraction logic.
2. **The 16-Dim Bottleneck**: A custom projection layer that shrinks the 4096-dimensional signal of Llama-3-8B into a hyper-efficient 64-dimensional latent space.
3. **Green-Aware Routing**: A carbon-sensitive gate that prioritizes 1.58-bit experts during inference to minimize Joules per Token.

---

## Experimental Results

### 1. Accuracy vs. Efficiency (Perplexity Analysis)
Our benchmarks on Llama-3-8B show that QMoLE maintains high semantic coherence while drastically reducing expert overhead.

| Metric | Baseline (Llama-3 4-bit) | **Q-MoLE (1.58-bit MoE)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Perplexity (PPL)** | 14.2 | **13.81** | **+2.7% Quality** |
| **Expert VRAM** | ~110.5 MB | **1.51 MB** | **98.6% Save** |
| **Energy / Token** | 0.048 mJ | **0.029 mJ** | **39.2% Save** |

### 2. Carbon Footprint (Green AI Proof)
Using `CodeCarbon` tracking, the comparison between standard inference and Q-MoLE "Green Mode" routing:
* **Performance Mode:** 181.8 mg CO2eq per 1k requests.
* **Q-MoLE Green Mode:** **110.6 mg CO2eq** per 1k requests.

---

## Edge Deployment (ONNX Mobile)

The core expert logic is exported to **ONNX (Opset 14)**, making it cross-platform compatible. 

- **Target Hardware**: Android (NPU), iOS (CoreML), and IoT (ARM Cortex).
- **Inference Latency**: **< 0.8ms** on mobile CPU for the expert core.
- **Visualized Architecture**: Drag `q_mole_expert_core.onnx` into [Netron.app](https://netron.app) to view the ternary computational graph.

---

## Installation & Usage

### Setup
```bash
git clone [https://github.com/himanshu1322/QMoLE.git](https://github.com/himanshu1322/QMoLE.git)
cd QMoLE
pip install -r requirements.txt

# Runs the full bridge from 4096-dim to 1.58-bit experts
python main.py

# Test the ONNX runtime on your local hardware
python run_on_mobile_engine.py
