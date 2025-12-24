# RAG-Chain
A RAG (Retrieval-Augmented Generation) framework that integrates LangChain, Ollama, and DeepSeek models, supporting unified invocation and deployment of multiple large language models.

# RAG-LangChain-Ollama-DeepSeek

模块化、可扩展的 RAG 框架，支持本地 Ollama 与 DeepSeek API 一键切换。

## 快速开始
1. 克隆仓库  
   ```bash
   git clone https://github.com/yourname/rag-langchain-ollama-deepseek.git
2. 安装依赖
cd rag-langchain-ollama-deepsee
pip install -r requirements.txt
3.启动本地 Ollama 并拉取模型
ollama pull llama3.1
4.配置 DeepSeek（可选）
将 DEEPSEEK_API_KEY=sk-xxx 写入 .env 文件。
5.放入文档
把任意 .txt 文件放到 data/sample.txt。
6.运行
python -m rag.cli -m ollama/python -m rag.cli -m deepseek
## 核心特性（含后续要完善的）
模块化设计：config / models / utils / chain 分层
多模型：Ollama 本地 & DeepSeek API 统一接口
配置集中：ConfigManager 单例
工具函数：哈希判重、格式化、日志
异步：DeepSeek 采用 AsyncOpenAI
依赖清晰：requirements.txt 一键安装
