from __future__ import annotations

import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from cli.generate import generate_text, load_model
from utils import get_device, get_logger

log = get_logger("lit-lm-server")

app_state: dict[str, object] = {}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_len: int = Field(200, ge=1, le=1024)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: int = Field(40, ge=0, le=200)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    output: str


def create_app(args: argparse.Namespace) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        device = get_device(args.device)
        log.info(f"Server using device: {device}")
        model, tokenizer = load_model(
            checkpoint_path=args.checkpoint,
            tokenizer_path=args.tokenizer,
            device=device,
            lora_path=args.lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
        app_state["model"] = model
        app_state["tokenizer"] = tokenizer
        app_state["device"] = device
        yield
        app_state.clear()

    app = FastAPI(
        title="Lit Language Model Server",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        output = generate_text(
            model=app_state["model"],
            tokenizer=app_state["tokenizer"],
            prompt=req.prompt,
            device=app_state["device"],
            max_len=req.max_len,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
        return GenerateResponse(output=output)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP inference server for the trained Transformer model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft/full/qa-decoder-only/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizers/wiki-pretrain-v1/tokenizer.model")
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
