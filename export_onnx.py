import torch
import torch.nn as nn
from models.moe import QMoLE_Expert # Export the Expert, not the whole Layer

def export_expert_only():
    hidden_dim = 64
    # We export the Expert specifically to show the 1.58-bit implementation
    expert = QMoLE_Expert(hidden_size=hidden_dim, intermediate_size=16)
    
    # Load your weights
    try:
        # We look for the weights of the first expert in your saved file
        full_weights = torch.load('weights/q_mole_experts_1_58bit.pth', map_location='cpu', weights_only=True)
        # Extract just Expert 0's weights
        expert_weights = {k.replace('experts.0.', ''): v for k, v in full_weights.items() if 'experts.0.' in k}
        expert.load_state_dict(expert_weights)
        print("✅ Expert-0 weights loaded.")
    except Exception as e:
        print(f"⚠️ Loading weights failed ({e}), exporting architecture.")

    expert.eval()
    dummy_input = torch.randn(1, 1, hidden_dim)

    torch.onnx.export(
        expert, 
        dummy_input, 
        "q_mole_expert_core.onnx",
        export_params=True,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {1: 'seq'}, 'output': {1: 'seq'}}
    )
    print("🚀 SUCCESS: q_mole_expert_core.onnx created!")

if __name__ == "__main__":
    export_expert_only()