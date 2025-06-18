import os
import asyncio
from typing import AsyncGenerator, Optional, List, Dict, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import uvicorn

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError("Missing google-genai. Install with 'pip install google-genai'") from e


class GeminiAgent:
    """Agent powered by Gemini models."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        *,
        api_key: Optional[str] = None,
        safety_settings: Optional[Union[List[genai.types.SafetySetting], List[Dict]]] = None,
        generation_config: Optional[dict] = None,
        # Add a system_instruction argument
        system_instruction: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY environment variable.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.safety_settings = safety_settings or []
        self.generation_config = generation_config or {}
        # Store the system instruction
        self.system_instruction = system_instruction


    def generate_response(self, query: str) -> str:
        """Return full response (blocking)."""
        gen_config = types.GenerateContentConfig(
            **self.generation_config,
            safety_settings=self.safety_settings
        )

        # Prepare the contents, including the system instruction if it exists
        contents = []
        if self.system_instruction:
            contents.append(self.system_instruction) # System instruction is usually the first part of contents
        contents.append(query) # User query follows


        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=contents, # Pass the combined contents
            config=gen_config,
        )
        return getattr(resp, "text", str(resp))

    async def generate_response_stream(self, query: str) -> AsyncGenerator[str, None]:
        """Stream response tokens asynchronously."""

        gen_config = types.GenerateContentConfig(
            **self.generation_config,
            safety_settings=self.safety_settings
        )

        # Prepare the contents, including the system instruction if it exists
        contents = []
        if self.system_instruction:
            contents.append(self.system_instruction) # System instruction
        contents.append(query) # User query

        def _blocking_stream():
            for chunk in self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents, # Pass the combined contents
                config=gen_config,
            ):
                yield chunk.text

        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, lambda: list(_blocking_stream()))
        for token in chunks:
            yield token


# Instantiate global agent with the desired system instruction
_agent = GeminiAgent(
    system_instruction="You are a helpful assistant who answer questions about F1."
)

app = FastAPI(
    title="Gemini Agent API",
    description="FastAPI wrapper around Google Gemini models (sync & SSE endpoints)",
    version="0.2.0",
)


class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None # Context is not used in the current Gemini API calls, but kept for schema


class QueryResponse(BaseModel):
    response: str


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is operational"}


@app.post("/agent", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    try:
        answer = _agent.generate_response(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(response=answer)


@app.post("/agent/stream")
async def query_agent_stream(request: QueryRequest):
    async def event_generator():
        try:
            async for chunk in _agent.generate_response_stream(request.query):
                yield {"data": chunk}
        except Exception as exc:
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)