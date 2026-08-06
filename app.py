import sys
import os

# ---------- 启动检测：直接 python 执行时自动转 streamlit run ----------
# 必须放在所有 streamlit 调用之前，否则会刷一堆 ScriptRunContext warning
if __name__ == "__main__":
    in_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        pass

    if not in_streamlit:
        import subprocess
        print("⚠️  检测到直接用 python 执行 Streamlit 应用，自动切换到 streamlit run ...\n")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)]
            + sys.argv[1:]
        )
        sys.exit(0)

import streamlit as st
import requests

# ---------- 页面配置 ----------
st.set_page_config(page_title="IT 运维知识库助手", layout="wide")
st.title("🛠️ IT 运维知识库助手")

# ---------- 侧边栏: 模式选择 ----------
with st.sidebar:
    st.header("⚙️ 设置")
    mode = st.radio(
        "问答模式",
        ["标准 RAG", "多智能体"],
        help="标准 RAG: 检索+重排序+生成\n多智能体: 诊断→方案→验证"
    )
    use_multi_query = st.checkbox("启用多查询重写", value=True)
    st.divider()
    st.caption("API: http://localhost:8000")
    st.caption("监控: http://localhost:9090")
    st.caption("Grafana: http://localhost:3000")

# ---------- 初始化聊天历史 ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是IT运维知识库助手，有什么可以帮你的？"}
    ]

# ---------- 显示聊天记录 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "trace" in msg:
            with st.expander("🔍 智能体执行轨迹"):
                for t in msg["trace"]:
                    st.json(t)

# ---------- 接收用户输入 ----------
if prompt := st.chat_input("请输入你的问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                if mode == "多智能体":
                    response = requests.post(
                        "http://localhost:8000/agent/chat",
                        json={"question": prompt, "mode": "agent"},
                        timeout=120
                    )
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        trace = data.get("agents_trace", [])
                        st.markdown(answer)
                        if trace:
                            with st.expander("🔍 智能体执行轨迹"):
                                for t in trace:
                                    st.json(t)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "trace": trace,
                        })
                    else:
                        error_msg = f"❌ API 错误 (状态码 {response.status_code})"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    response = requests.post(
                        "http://localhost:8000/chat",
                        json={"question": prompt, "use_multi_query": use_multi_query},
                        timeout=60
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
