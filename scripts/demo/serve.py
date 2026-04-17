"""
Inference server — wraps ObfuscationPipeline behind a FastAPI HTTP endpoint.
Consumed by the Go server (server/) via POST /infer.

Usage:
    pip install fastapi uvicorn
    python scripts/demo/serve.py
    python scripts/demo/serve.py --port 8000 --mode placeholder
"""

import argparse
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from obfuscate import ObfuscationPipeline, DEMO_USERS


pipeline: ObfuscationPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    mode = app.state.mode
    pipeline = ObfuscationPipeline(DEMO_USERS, mode=mode)
    print(f"pipeline ready  mode={mode}", file=sys.stderr)
    yield


app = FastAPI(lifespan=lifespan)


class InferRequest(BaseModel):
    text: str
    session_id: str = "default"


@app.post("/infer")
def infer(req: InferRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")
    result = pipeline.process(req.text, session_id=req.session_id)
    detected = [
        {**{k: float(v) if hasattr(v, "item") else v for k, v in d.items()}}
        for d in result["detected"]
    ]
    return {
        "text": result["obfuscated"],
        "detected": detected,
        "used_model": result["used_model"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mode", choices=["fakename", "placeholder", "redact"], default="fakename")
    args = parser.parse_args()

    app.state.mode = args.mode
    uvicorn.run(app, host=args.host, port=args.port)
