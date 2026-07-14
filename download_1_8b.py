"""下载 InternLM2-1.8B 嬛嬛模型"""
from modelscope import snapshot_download

model_id = "kmno4zx/huanhuan-chat-internlm2-1_8b"
print(f"正在下载模型: {model_id} ...")
print(f"大小约 3.6GB，请耐心等待...")

local_path = snapshot_download(model_id, revision="master", cache_dir="./")
print(f"\n✅ 下载完成！模型路径: {local_path}")
