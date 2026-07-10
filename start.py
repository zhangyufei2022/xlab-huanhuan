import os
import subprocess

# 解决 Windows 控制台中文乱码问题
os.environ["PYTHONIOENCODING"] = "utf-8"

# 使用 subprocess 代替 os.system，确保编码正确
subprocess.run(
    ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port", "7860"],
    encoding="utf-8",
    errors="replace",
)