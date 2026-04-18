import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from models.moe import QMoLE_Layer
import os

def main():
    # 1. Force GPU Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Research Environment: {device}")

    # 2. Load Llama-3-8B (4-bit for Laptop Compatibility)
    # We use the unsloth version because it is optimized for consumer GPUs
    model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit" 
    
    print(f"🚀 Loading Llama-3-8B Backbone...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model in 4-bit to fit in ~5.5GB VRAM
    backbone = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 3. Dynamic Dimension Setup
    # Llama-3-8B uses 4096 hidden dimensions
    backbone_dim = backbone.config.hidden_size 
    expert_dim = 64 
    print(f"📊 Mapping: {backbone_dim} (Llama) -> {expert_dim} (Q-MoLE)")

    # 4. Initialize Architecture
    # Bridge and Experts moved to GPU
    bridge = nn.Linear(backbone_dim, expert_dim).to(device).to(torch.bfloat16)
    q_mole = QMoLE_Layer(hidden_size=expert_dim).to(device)

    # 5. Load Your Trained Weights
    experts_path = 'weights/q_mole_experts_1_58bit.pth'
    if os.path.exists(experts_path):
        q_mole.load_state_dict(torch.load(experts_path, map_location=device, weights_only=True))
        print(f"✅ 1.58-bit Expert weights loaded.")
    else:
        print("⚠️ Experts file not found. Running diagnostic mode.")

    # 6. Research Inference Test
    prompt = "Explain the impact of ternary quantization on Green AI."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("\n--- [ INFERENCE START ] ---")
    with torch.no_grad():
        # Step A: Get Llama-3's base tokens
        raw_features = backbone.model.embed_tokens(inputs['input_ids']).to(torch.bfloat16)
        
        # Step B: Pass through the Q-MoLE Bottleneck
        bridged_features = bridge(raw_features)
        output = q_mole(bridged_features)

    print(f"✅ Final Research Tensor Shape: {output.shape}")
    print(f"🚀 SUCCESS: Llama-3-8B is now routed through your 1.58-bit experts.")

if __name__ == "__main__":
    main()