import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.moe import QMoLE_Layer
import os

def main():
    # 1. Force GPU Usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device detected: {device}")

    # 2. Load Backbone (TinyLlama)
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    backbone = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to(device)

    # 3. Setup Q-MoLE Architecture
    # The bridge connects Llama (2048) to your experts (64)
    bridge = nn.Linear(backbone.config.hidden_size, 64).to(device)
    q_mole = QMoLE_Layer(hidden_size=64).to(device)

    # 4. Loading Logic
    bridge_path = 'weights/adapter_bridge.pth'
    experts_path = 'weights/q_mole_experts_1_58bit.pth'

    # Load Experts if they exist
    if os.path.exists(experts_path):
        q_mole.load_state_dict(torch.load(experts_path, map_location=device))
        print(f"✅ Expert weights loaded from {experts_path}")
    else:
        print("⚠️ Experts file not found. Using random experts.")

    # Load Bridge if it exists, otherwise skip
    if os.path.exists(bridge_path):
        bridge.load_state_dict(torch.load(bridge_path, map_location=device))
        print(f"✅ Bridge weights loaded from {bridge_path}")
    else:
        print("ℹ️ Bridge file not found. Initializing with random weights for demo.")

    # 5. Inference Test
    prompt = "Explain the efficiency of ternary quantization."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get embeddings from Llama
        embeddings = backbone.model.embed_tokens(inputs['input_ids']).to(torch.float32)
        
        # Pass through the bridge then the experts
        bridged_features = bridge(embeddings)
        output = q_mole(bridged_features)

    print(f"🚀 SUCCESS! Final Output Shape: {output.shape}")

if __name__ == "__main__":
    main()