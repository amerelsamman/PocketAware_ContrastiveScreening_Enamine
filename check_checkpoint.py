"""Quick script to verify checkpoint contents."""
import torch

# Check Stage 2 checkpoint
ckpt = torch.load('models/stage2_best.pt')

print("Stage 2 Checkpoint Contents:")
print("=" * 50)
print(f"Keys: {list(ckpt.keys())}")
print(f"Epoch: {ckpt['epoch']}")
print(f"Val AUC: {ckpt['val_auc']:.4f}")
print(f"Model state dict: {len(ckpt['model_state_dict'])} parameter tensors")
print(f"Optimizer state dict: {'optimizer_state_dict' in ckpt}")

print("\nModel parameter shapes (first 5):")
for i, (name, param) in enumerate(list(ckpt['model_state_dict'].items())[:5]):
    print(f"  {name}: {param.shape}")

print("\nCheckpoint verified ✓")
print("Ready for inference on Enamine compounds.")
