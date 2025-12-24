import asyncio
from typing import List
import ollama
from openai import AsyncOpenAI
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field
from rag.config import config

class OllamaLLM(LLM):
    model: str = Field(default=config.ollama.model)

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
    ) -> str:
        resp = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.7, "stop": stop or []},
        )
        return resp["response"]

    @property
    def _llm_type(self) -> str:
        return "ollama"

class DeepSeekLLM(LLM):
    client: AsyncOpenAI = Field(
        default_factory=lambda: AsyncOpenAI(
            api_key=config.deepseek.api_key,
            base_url=config.deepseek.base_url,
        )
    )
    model: str = Field(default=config.deepseek.model)

    def _call(
        self,
        prompt: str,
        stop: List[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
    ) -> str:
        # 同步包装异步
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._acall(prompt, stop))

    async def _acall(
        self, prompt: str, stop: List[str] | None = None
    ) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stop=stop or [],
        )
        return resp.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "deepseek"
