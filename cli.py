import argparse
from rag.chain import RAGChain

def main():
    parser = argparse.ArgumentParser(description="RAG with Ollama / DeepSeek")
    parser.add_argument("-m", "--model", choices=["ollama", "deepseek"], default="ollama")
    args = parser.parse_args()

    chain = RAGChain(model_type=args.model)
    print("RAG 系统已加载，输入 exit 退出。")
    while True:
        q = input("\n>>> Question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print(chain.ask(q))

if __name__ == "__main__":
    main()
