import torch
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
x = torch.rand(1000, 1000, device="cuda")
print("Test calculation OK:", (x @ x).shape)
