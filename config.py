import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"          # 本地模型名
    timeout: int = 60

@dataclass
class DeepSeekConfig:
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY"))
    model: str = "deepseek-chat"

@dataclass
class ChromaConfig:
    persist_dir: str = "./chroma_db"
    collection: str = "rag_docs"
    chunk_size: int = 800
    chunk_overlap: int = 100

@dataclass
class Config:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)

# 全局单例
config = Config()
