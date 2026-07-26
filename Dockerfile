# 使用 Python 3.10 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# 安装系统依赖（用于支持 sentence-transformers 等）
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（排除 .dockerignore 中的文件）
COPY . .

# 创建用于存放模型缓存的目录
RUN mkdir -p /root/.cache/huggingface

# 暴露端口
EXPOSE 8000
EXPOSE 8501

# 安装 supervisor 来同时运行两个服务
RUN pip install supervisor

# 创建 supervisor 配置文件（使用 python -m 方式）
RUN echo '[supervisord]\n\
nodaemon=true\n\
\n\
[program:fastapi]\n\
command=python -m uvicorn api:app --host 0.0.0.0 --port 8000\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/var/log/fastapi.err.log\n\
stdout_logfile=/var/log/fastapi.out.log\n\
\n\
[program:streamlit]\n\
command=python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/var/log/streamlit.err.log\n\
stdout_logfile=/var/log/streamlit.out.log\n' > /etc/supervisord.conf

# 暴露端口
EXPOSE 8000
EXPOSE 8501

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisord.conf"]
