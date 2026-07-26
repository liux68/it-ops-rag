import streamlit as st
import requests

# ---------- 页面配置 ----------
st.set_page_config(page_title="IT 运维知识库助手", layout="centered")
st.title("🛠️ IT 运维知识库助手")

# ---------- 初始化聊天历史 ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是IT运维知识库助手，有什么可以帮你的？"}
    ]

# ---------- 显示聊天记录 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- 接收用户输入 ----------
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用后端 API
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                # 如果API与前端在同一台机器，地址为 http://localhost:8000
                # 如果部署到服务器，请修改为实际的API地址
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={"question": prompt},
                    timeout=30
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"❌ API 错误 (状态码 {response.status_code})"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到后端服务，请确保 FastAPI 已启动 (python api.py)")
            except Exception as e:
                st.error(f"❌ 发生错误: {e}")