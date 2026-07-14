# Python 3.12+ 兼容性补丁：恢复已移除的 ast 属性
import ast
for _attr in ("Str", "Num", "Bytes", "NameConstant", "Ellipsis"):
    if not hasattr(ast, _attr):
        setattr(ast, _attr, ast.Constant)

# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import os

# 检测设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU 推理必须用 float32，float16 在 x86 CPU 上非常慢（无原生支持）
device_dtype = torch.bfloat16 if device == "cuda" else torch.float32

# CPU 线程优化：使用多核加速推理
if device == "cpu":
    num_threads = os.cpu_count() or 8  # 不设上限，全核投入
    torch.set_num_threads(num_threads)
    # 同时设置 OpenMP/MKL 线程数（部分 PyTorch 版本需要手动设）
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    print(f"[CPU 优化] 已设置 {num_threads} 个线程用于推理")

# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## InternLM LLM")
    st.caption(f"当前运行设备: **{device.upper()}** {'⚠️ CPU 速度较慢，请耐心等待' if device == 'cpu' else ''}")
    "[InternLM](https://github.com/InternLM/InternLM.git)"
    "[开源大模型食用指南 self-llm](https://github.com/datawhalechina/self-llm.git)"
    "[Chat嬛嬛](https://github.com/KMnO4-zx/huanhuan-chat.git)"
    # 创建一个滑块，用于选择最大生成长度，默认值降低到256以加快CPU响应
    max_length = st.slider("max_new_tokens", 32, 1024, 256, step=32,
                           help="每次回复最多生成的 token 数量，值越小速度越快")
    system_prompt = st.text_input("System_Prompt", "现在你要扮演皇帝身边的女人--甄嬛")

# 创建一个标题和一个副标题
st.title("💬 InternLM2-Chat-1.8B 嬛嬛版")
st.caption("🚀 A streamlit chatbot powered by InternLM2 QLora")

# 定义模型路径

# 本地模型路径（已下载完成）
# # 本地模型路径（1.8B 版本，CPU 推理更快；如需切回 7B，改为下面注释的路径）
mode_name_or_path = "./models/kmno4zx--huanhuan-chat-internlm2-1_8b/snapshots/master"
# mode_name_or_path = "./models/kmno4zx--huanhuan-chat-internlm2/snapshots/master"  # 7B 版本

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    # 从预训练的模型中获取tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    # 从预训练的模型中获取模型，并设置模型参数
    model = AutoModelForCausalLM.from_pretrained(
        mode_name_or_path,
        trust_remote_code=True,
        torch_dtype=device_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model

# 加载模型（首次加载较慢，后续 Streamlit rerun 会使用缓存）
with st.spinner("正在加载模型，请耐心等待..."):
    tokenizer, model = get_model()

# 如果session_state中没有"messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message("user").write(msg[0])
    st.chat_message("assistant").write(msg[1])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)

    # 使用流式输出生成回复（逐token显示，提升CPU模式下的用户体验）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("思考中..."):
            response_text = ""
            # stream_chat 是生成器，每生成一个 token 就会 yield 一次
            for chunk, history in model.stream_chat(
                tokenizer,
                prompt,
                history=st.session_state.messages,
                meta_instruction=system_prompt,
                max_new_tokens=max_length,
            ):
                response_text = chunk
                message_placeholder.markdown(response_text + "▌")
        # 去掉光标符号，显示最终结果
        message_placeholder.markdown(response_text)

    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append((prompt, response_text))