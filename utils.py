import hashlib
import time
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from rag.config import config

def doc_hash(file_path: str | Path) -> str:
    """生成文档哈希，用于判断是否需要重新索引"""
    h = hashlib.sha256()
    h.update(Path(file_path).read_bytes())
    return h.hexdigest()[:16]

def format_source(doc) -> str:
    """统一格式化检索结果"""
    return f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content[:150]}..."

def log_query(query: str, answer: str, duration: float):
    """简单日志，可换成 loguru 或持久化到文件"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Q: {query} | T: {duration:.2f}s\nA: {answer}\n")

def build_or_load_vectorstore(
    file_path: str | Path, force_rebuild: bool = False
) -> Chroma:
    """如果哈希变化或强制重建则重新索引"""
    file_path = Path(file_path)
    hash_file = file_path.with_suffix(".hash")
    current_hash = doc_hash(file_path)

    if not force_rebuild and hash_file.exists() and hash_file.read_text() == current_hash:
        print("VectorStore 已存在且文档无变化，直接加载...")
        return Chroma(
            persist_directory=config.chroma.persist_dir,
            embedding_function=OllamaEmbeddings(
                model=config.ollama.model, base_url=config.ollama.base_url
            ),
        )

    print("重新构建 VectorStore...")
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chroma.chunk_size,
        chunk_overlap=config.chroma.chunk_overlap,
    )
    texts = splitter.split_documents(docs)

    db = Chroma.from_documents(
        documents=texts,
        embedding=OllamaEmbeddings(
            model=config.ollama.model, base_url=config.ollama.base_url
        ),
        persist_directory=config.chroma.persist_dir,
        collection_name=config.chroma.collection,
    )
    db.persist()
    hash_file.write_text(current_hash)
    return db
