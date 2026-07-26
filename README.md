# it-ops-rag
# 🛠️ IT 运维 RAG 知识库智能问答系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0-green.svg)](https://www.langchain.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-✅-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **从零构建的企业级 RAG 知识库系统** | 混合检索 + 重排序 + API 封装 + 容器化部署

---

## 📖 项目简介

企业内部文档和知识库快速增长，运维人员在大量资料中定位故障解决方案耗时严重。本项目基于 **检索增强生成（RAG）** 技术，构建了一套端到端的智能问答系统。用户通过自然语言提问，系统自动从知识库中检索相关内容并生成准确回答，**将平均问题定位时间从 10 分钟缩短至 10 秒**。

### 🎯 核心价值

- **智能检索**：理解自然语言语义，精准定位相关知识片段
- **混合召回**：结合关键词匹配和语义理解，提升召回率和精准度
- **来源可溯**：每个回答均附文档来源，确保答案可信可查
- **开箱即用**：支持 Docker 一键部署，5 分钟完成环境搭建
- **接口开放**：提供 RESTful API，可轻松集成至企业微信、钉钉等平台

---

## ✨ 核心特性

| 模块 | 技术方案 | 说明 |
|------|----------|------|
| **文档处理** | 多格式加载（PDF/TXT/Markdown）+ 语义分块 | 支持按段落和语义边界切分，保持知识单元完整性 |
| **混合检索** | BM25 关键词匹配 + FAISS 向量检索 | 双路召回 + RRF 融合，兼顾精确匹配和语义理解 |
| **重排序** | Cross-Encoder 精排模型 | 对初筛结果进行深度排序，将最相关文档排至前列 |
| **问答生成** | LangChain LCEL + DeepSeek/OpenAI | 基于检索结果生成精准、可控的回答 |
| **API 服务** | FastAPI 异步接口 | 提供 `/chat`、`/health` 等 RESTful 接口 |
| **Web 界面** | Streamlit 交互式聊天 | 开箱即用的对话界面，支持连续多轮问答 |
| **容器化** | Docker + Docker Compose | 一键部署，环境一致性有保障 |

---

## 🛠️ 技术栈

| 分类 | 技术 | 说明 |
|------|------|------|
| **框架** | LangChain 0.3.x | RAG 应用开发核心框架，使用 LCEL 构建链式流程 |
| **向量数据库** | FAISS | Meta 开源，轻量级本地向量索引，支持数据隐私 |
| **检索算法** | BM25 + 向量检索 | 混合检索架构，兼顾关键词精确匹配与语义相似度 |
| **重排序** | Cross-Encoder | 对初筛结果精排，提升最终答案质量 |
| **嵌入模型** | sentence-transformers/all-MiniLM-L6-v2 | 轻量高效，支持本地部署 |
| **大语言模型** | DeepSeek / OpenAI 兼容接口 | 云端调用，可切换至本地模型 |
| **API 框架** | FastAPI + Uvicorn | 异步高性能 Web 服务，自动生成 Swagger 文档 |
| **前端界面** | Streamlit | 快速构建交互式聊天界面 |
| **容器化** | Docker + Docker Compose | 一键部署，环境隔离 |
| **版本控制** | Git + GitHub | 代码托管与版本管理 |

---

## 🚀 快速开始

### 前置条件

- Python 3.10+（本地运行）或 Docker Desktop（容器化部署）
- 有效的 LLM API Key（DeepSeek / OpenAI 等）

---

### 方式一：Docker 部署（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/liux68/it-ops-rag.git
cd it-ops-rag

# 2. 配置 API Key（编辑 .env 文件）
cp .env.example .env
# 填入你的 OPENAI_API_KEY

# 3. 构建并启动
docker-compose build
docker-compose up -d

# 4. 访问服务
# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000/docs
