import os
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path
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

    def execute_code(self, code: str) -> Dict[str, str]:
        """Execute Python code in a sandboxed environment and return stdout/stderr."""
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add FastF1 cache setup to every execution
                # Indent the user code properly
                indented_code = '\n'.join('    ' + line for line in code.split('\n'))
                full_code = f"""import sys
import os
sys.path.insert(0, os.getcwd())

try:
    import fastf1
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Enable FastF1 caching (create directory if it doesn't exist)
    import os
    cache_dir = "/tmp/fastf1_cache"
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    
{indented_code}
    
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    import traceback
    traceback.print_exc()
"""
                f.write(full_code)
                f.flush()
                
                # Execute the code
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=60,  # 60 second timeout
                    cwd=os.getcwd()
                )
                
                # Clean up
                os.unlink(f.name)
                
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Code execution timed out", "return_code": 1}
        except Exception as e:
            return {"stdout": "", "stderr": f"Execution error: {e}", "return_code": 1}

    def _process_react_response(self, response: str) -> str:
        """Process ReAct response and execute code blocks marked with <EXECUTE>."""
        import re
        
        # Find code blocks followed by <EXECUTE>
        code_pattern = r'```python\n(.*?)\n```\s*<EXECUTE>'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        processed_response = response
        
        for code_block in matches:
            # Execute the code
            execution_result = self.execute_code(code_block)
            
            # Format the execution result
            result_section = "\n\n### Result\n"
            if execution_result["stdout"]:
                result_section += execution_result["stdout"]
            if execution_result["stderr"]:
                result_section += f"\nErrors: {execution_result['stderr']}"
            if execution_result["return_code"] != 0:
                result_section += f"\nExit code: {execution_result['return_code']}"
            
            # Replace <EXECUTE> with the result
            processed_response = processed_response.replace(
                f"```python\n{code_block}\n```\n<EXECUTE>",
                f"```python\n{code_block}\n```\n<EXECUTE>{result_section}"
            )
        
        return processed_response


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
        response_text = getattr(resp, "text", str(resp))
        
        # Process ReAct workflow with code execution
        return self._process_react_response(response_text)

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


FASTF1_SYSTEM_INSTRUCTION = """
You are "FastF1-Agent", an autonomous Python assistant that answers Formula 1 questions by planning, coding, executing, and explaining with the FastF1 library. 
Your knowledge cutoff isn't limited to 2025. With FastF1 Library you get access to most up to date data by querying it with the fastf1 library. 
You follow the ReAct pattern to interleave reasoning with actions.

### Required Workflow Template
For EVERY user request, respond in EXACTLY this order:

### Thought
1. [Short, numbered reasoning steps]
2. [Break down the F1 question into sub-tasks]

### Code
```python
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
fastf1.Cache.enable_cache("/tmp/fastf1_cache")

# Your FastF1 logic here
# Use patterns like:
# session = fastf1.get_session(year, event, session_type)
# session.load()
# Print results with print(df.to_string(index=False))
```

<EXECUTE>

### Result
[Echo stdout/stderr from execution]

### Answer
[Natural-language conclusion based on results]

### FastF1 Usage Patterns:
- Session loading: `fastf1.get_session(2023, "Monaco", "Q")` 
- Always enable caching: `fastf1.Cache.enable_cache("/tmp/fastf1_cache")`
- Load data: `session.load()` or selective loading with parameters
- Driver filtering: `session.laps.pick_drivers('VER')`
- Fastest laps: `session.laps.pick_fastest()`
- Output tables: `print(df.to_string(index=False))`
- Visualizations: Use matplotlib, describe plots as "[Plot generated: description]"

### Safety Rules:
- Code must be self-contained Python 3.11+ compatible
- Never fabricate data, only use FastF1 APIs
- Handle exceptions gracefully
- Keep tables ≤20 rows for readability
- Place <EXECUTE> marker after code blocks

Never place the answer before reasoning or code execution.
"""

_agent = GeminiAgent(
    system_instruction=FASTF1_SYSTEM_INSTRUCTION
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


@app.post("/chat", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    try:
        answer = _agent.generate_response(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(response=answer)


@app.post("/chat/stream")
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