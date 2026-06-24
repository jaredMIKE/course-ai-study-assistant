from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from . import config
from .database import (
    init_db,
    add_document,
    list_documents,
    delete_document,
    stats as db_stats,
    clear_and_seed,
    retrieve,
    db,
    all_document_text,
    save_history,
    history as db_history,
)
from .llm import (
    rag_answer,
    llm_available,
    llm_summary,
    llm_quiz,
    llm_flashcards,
    llm_review_plan,
    fallback_summary,
    fallback_quiz,
    fallback_flashcards,
    fallback_review_plan,
)
from .nlp import keywords
from .parsers import extract_text_from_file
from .schemas import DocumentIn, ChatIn

app = FastAPI(title=config.APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in __import__("os").getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    index_path = config.FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})
    return HTMLResponse("<h1>专业课程资料辅助学习系统 API 已启动</h1><p>前端文件不存在，请检查 frontend/index.html。</p>")


@app.get("/api")
def api_root() -> dict[str, str]:
    return {"name": config.APP_TITLE, "status": "running"}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_enabled": llm_available(),
        "model": config.LLM_MODEL,
        "base_url": config.LLM_BASE_URL,
        "supported_files": config.SUPPORTED_EXTENSIONS,
        "db_path": str(config.DB_PATH),
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    return db_stats()


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    return {"items": list_documents()}


@app.post("/api/documents")
def create_document(payload: DocumentIn) -> dict[str, Any]:
    with db() as conn:
        result = add_document(conn, payload.title, payload.text, source_type="paste")
    return {"message": "资料已加入知识库", **result}


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: int) -> dict[str, Any]:
    ok = delete_document(document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"message": "文档已删除", "id": document_id}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="请选择至少一个文件")
    results = []
    with db() as conn:
        for file in files:
            raw = await file.read()
            safe_name, text = extract_text_from_file(file.filename or "上传资料", raw)
            ext = Path(safe_name).suffix.lower()
            result = add_document(
                conn,
                title=Path(safe_name).stem or safe_name,
                text=text,
                source_type="upload",
                file_name=safe_name,
                file_ext=ext,
            )
            results.append({"filename": safe_name, **result})
    return {"message": f"成功上传 {len(results)} 个文件", "items": results}


@app.delete("/api/documents")
def clear_documents() -> dict[str, str]:
    clear_and_seed()
    return {"message": "知识库已重置为示例资料"}


@app.post("/api/chat")
async def chat(payload: ChatIn) -> dict[str, Any]:
    sources = retrieve(payload.question, payload.top_k, payload.threshold)
    if not sources:
        return {"answer": "知识库为空，请先上传或粘贴课程资料。", "sources": [], "confidence": 0, "used_llm": False}
    if payload.use_llm:
        answer, used_llm, llm_error = await rag_answer(payload.question, sources, payload.mode)
    else:
        from .llm import template_answer

        answer = template_answer(payload.question, sources, payload.mode)
        used_llm, llm_error = False, "用户关闭 LLM"
    save_history(payload.question, answer, sources, used_llm, llm_error)
    confidence = max([source["score"] for source in sources], default=0.0)
    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence * 100, 1),
        "used_llm": used_llm,
        "llm_error": llm_error,
    }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1, description="检索关键词或问题"),
    top_k: int = Query(8, ge=1, le=20),
    threshold: float = Query(0.0, ge=0, le=1),
) -> dict[str, Any]:
    items = retrieve(q, top_k=top_k, threshold=threshold)
    return {"query": q, "items": items}


@app.get("/api/summary")
async def summary(use_llm: bool = True) -> dict[str, Any]:
    text = all_document_text()
    if not text.strip():
        return {"points": [], "keywords": [], "used_llm": False, "content": "知识库为空。"}
    if use_llm:
        content, used, error = await llm_summary(text)
        if content:
            return {"content": content, "used_llm": used, "llm_error": error, "keywords": keywords(text, 14)}
    fallback = fallback_summary(text)
    return {**fallback, "used_llm": False, "llm_error": None if not use_llm else "未配置或调用 LLM 失败，已使用本地摘要"}


@app.get("/api/quiz")
async def quiz(use_llm: bool = True) -> dict[str, Any]:
    text = all_document_text()
    sources = retrieve("课程重点 知识点 算法 复习 测验", top_k=8, threshold=0)
    if use_llm and text.strip():
        content, used, error = await llm_quiz(text)
        if content:
            return {"content": content, "used_llm": used, "llm_error": error, "items": []}
    return {"items": fallback_quiz(sources), "used_llm": False, "llm_error": None if not use_llm else "未配置或调用 LLM 失败，已使用本地出题"}


@app.get("/api/flashcards")
async def flashcards(use_llm: bool = True) -> dict[str, Any]:
    text = all_document_text()
    sources = retrieve("核心概念 重点 流程 算法 易错点", top_k=10, threshold=0)
    if use_llm and text.strip():
        content, used, error = await llm_flashcards(text)
        if content:
            return {"content": content, "used_llm": used, "llm_error": error, "items": []}
    return {"items": fallback_flashcards(sources), "used_llm": False, "llm_error": None if not use_llm else "未配置或调用 LLM 失败，已使用本地卡片"}


@app.get("/api/review-plan")
async def review_plan(use_llm: bool = True) -> dict[str, Any]:
    text = all_document_text()
    if not text.strip():
        return {"content": "知识库为空。", "used_llm": False, "items": []}
    if use_llm:
        content, used, error = await llm_review_plan(text)
        if content:
            return {"content": content, "used_llm": used, "llm_error": error}
    fallback = fallback_review_plan(text)
    return {**fallback, "used_llm": False, "llm_error": None if not use_llm else "未配置或调用 LLM 失败，已使用本地计划"}


@app.get("/api/keywords")
def keyword_overview() -> dict[str, Any]:
    text = all_document_text(limit_chars=20000)
    docs = list_documents()
    return {
        "keywords": keywords(text, 30) if text else [],
        "documents": [
            {"id": doc["id"], "title": doc["title"], "keywords": [k for k in str(doc.get("keywords") or "").split(",") if k]}
            for doc in docs
        ],
    }


@app.get("/api/history")
def history() -> dict[str, Any]:
    return {"items": db_history(30)}
