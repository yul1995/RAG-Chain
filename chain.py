import time
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rag.models import OllamaLLM, DeepSeekLLM
from rag.utils import build_or_load_vectorstore, log_query, format_source

prompt_tmpl = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the following context to answer the question.
Context:
{context}

Question: {question}
Answer ( concise, in Chinese ):""",
)

class RAGChain:
    def __init__(self, model_type: str = "ollama"):
        if model_type == "ollama":
            llm = OllamaLLM()
        elif model_type == "deepseek":
            llm = DeepSeekLLM()
        else:
            raise ValueError("model_type must be ollama or deepseek")

        vectorstore = build_or_load_vectorstore("data/sample.txt")
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt_tmpl},
            return_source_documents=True,
        )

    def ask(self, query: str) -> str:
        t0 = time.time()
        result = self.qa({"query": query})
        elapsed = time.time() - t0
        answer = result["result"]
        sources = "\n".join(format_source(doc) for doc in result["source_documents"])
        log_query(query, answer, elapsed)
        return f"{answer}\n\nSources:\n{sources}"
