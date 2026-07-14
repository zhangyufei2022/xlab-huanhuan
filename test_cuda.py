"""CUDA 环境检测脚本"""
import torch
import sys

print("=" * 50)
print("CUDA 环境检测")
print("=" * 50)

# 1. PyTorch 版本
print(f"\n[1] PyTorch 版本: {torch.__version__}")

# 2. CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"\n[2] CUDA 是否可用: {cuda_available}")

if not cuda_available:
    print("\n❌ CUDA 不可用，请检查：")
    print("  - 是否安装了 NVIDIA 显卡驱动")
    print("  - PyTorch 版本是否为 CUDA 版本（非 CPU-only）")
    print("\n  安装 CUDA 版 PyTorch：")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(0)

# 3. CUDA 版本
print(f"\n[3] PyTorch 编译时的 CUDA 版本: {torch.version.cuda}")
print(f"    驱动支持的 CUDA 版本: {torch.version.cuda_driver if hasattr(torch.version, 'cuda_driver') else 'N/A'}")

# 4. GPU 信息
gpu_count = torch.cuda.device_count()
print(f"\n[4] 可用 GPU 数量: {gpu_count}")
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"\n    GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"        显存总量: {props.total_memory / 1024**3:.1f} GB")
    print(f"        计算能力: {props.major}.{props.minor}")
    print(f"        多处理器数: {props.multi_processor_count}")

# 5. cuDNN
try:
    cudnn_version = torch.backends.cudnn.version()
    print(f"\n[5] cuDNN 版本: {cudnn_version}")
except:
    print(f"\n[5] cuDNN: 未安装")

# 6. 简单性能测试
print("\n[6] 运行简单计算测试...")
x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000, 1000, device="cuda")
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
z = x @ y
end.record()
torch.cuda.synchronize()
elapsed = start.elapsed_time(end)
print(f"    矩阵乘法 (1000x1000) 耗时: {elapsed:.2f} ms")

print("\n" + "=" * 50)
print("✅ CUDA 环境正常，GPU 可用！")
print("=" * 50)
