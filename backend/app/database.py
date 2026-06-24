from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import sqlite3
from typing import Any, Optional

from fastapi import HTTPException

from .config import DB_PATH
from .nlp import normalize, keywords, split_text, tfidf_vectors, vectorize_with_idf, cosine

DEFAULT_SAMPLE = """
人工智能方向综合实习课程要求学生完成一个完整应用系统。项目技术文档应包含需求分析、项目设计、关键算法设计、任务分工、项目实现、项目测试和项目总结。
专业课程资料辅助学习系统面向大学课程复习场景。课程资料通常包括 PPT 课件、PDF 讲义、参考书目、课堂录音转写文本和实验指导书。学生在期末复习时经常遇到资料零散、知识点关联弱、难以快速定位重点的问题。
本系统基于 RAG 检索增强生成思想构建课程辅助学习平台。系统首先对课程资料进行统一导入和文本清洗，然后将长文档切分为多个语义片段，使用轻量 TF-IDF 向量化和余弦相似度检索召回相关片段，最后结合大语言模型 API 生成有依据的回答。
RAG 的核心流程包括文档加载、文本清洗、语义切分、向量化、向量存储、相似度检索、提示词组装和答案生成。与直接让大模型回答相比，RAG 可以减少幻觉，提高回答与课程资料的一致性，并支持答案来源追溯。
系统功能模块包括资料解析模块、知识抽取模块、智能问答模块、溯源展示模块和测验生成模块。资料解析模块负责读取 TXT、PDF、Word、PPT 等课程资料并清洗无关符号。知识抽取模块自动总结章节要点、关键词和复习建议。智能问答模块根据用户问题进行检索并生成回答。溯源展示模块显示命中的原文片段和相似度。测验生成模块根据重点知识点生成判断题、简答题和应用题。
项目的关键算法包括文档分块算法、TF-IDF 关键词权重计算、余弦相似度检索、Top-K 排序和基于上下文的问答生成。文档分块算法通过固定长度和重叠窗口保留上下文。关键词权重用于衡量词语在文档片段中的重要程度。余弦相似度用于比较用户问题与知识库片段之间的相关程度。
系统测试应覆盖资料导入、知识库构建、检索准确性、问答生成、测验生成和页面交互。典型测试用例包括上传课程资料后是否能生成分块，提问 RAG 流程是否返回相关答案，生成的测验题是否来自知识库内容，清空知识库后系统是否正确提示。
""".strip()


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(
            """
            create table if not exists documents (
              id integer primary key autoincrement,
              title text not null,
              text text not null,
              source_type text not null default 'paste',
              file_name text,
              file_ext text,
              char_count integer not null default 0,
              keywords text not null,
              created_at text not null
            );
            create table if not exists chunks (
              id integer primary key autoincrement,
              document_id integer not null,
              document_title text not null,
              chunk_index integer not null,
              text text not null,
              keywords text not null,
              char_count integer not null default 0,
              created_at text not null,
              foreign key(document_id) references documents(id) on delete cascade
            );
            create table if not exists qa_history (
              id integer primary key autoincrement,
              question text not null,
              answer text not null,
              source_count integer not null,
              confidence real not null,
              used_llm integer not null,
              llm_error text,
              created_at text not null
            );
            """
        )
        count = conn.execute("select count(*) from documents").fetchone()[0]
        if count == 0:
            add_document(conn, "人工智能方向课程综合实习讲义", DEFAULT_SAMPLE, source_type="sample")


def add_document(
    conn: sqlite3.Connection,
    title: str,
    text: str,
    source_type: str = "paste",
    file_name: Optional[str] = None,
    file_ext: Optional[str] = None,
) -> dict[str, Any]:
    clean = normalize(text)
    if not clean:
        raise HTTPException(status_code=400, detail="资料内容不能为空")
    now = datetime.now().isoformat(timespec="seconds")
    title = (title or file_name or "未命名资料").strip()[:120]
    doc_keywords = ",".join(keywords(clean, 12))
    cursor = conn.execute(
        """
        insert into documents(title, text, source_type, file_name, file_ext, char_count, keywords, created_at)
        values(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (title, clean, source_type, file_name, file_ext, len(clean), doc_keywords, now),
    )
    doc_id = int(cursor.lastrowid)
    chunks = split_text(clean)
    if not chunks and clean:
        chunks = [clean]
    for index, chunk in enumerate(chunks, start=1):
        conn.execute(
            """
            insert into chunks(document_id, document_title, chunk_index, text, keywords, char_count, created_at)
            values(?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, title, index, chunk, ",".join(keywords(chunk, 10)), len(chunk), now),
        )
    return {"id": doc_id, "title": title, "chunk_count": len(chunks), "char_count": len(clean)}


def list_documents() -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute(
            """
            select d.id, d.title, d.source_type, d.file_name, d.file_ext, d.keywords, d.created_at,
                   d.char_count as text_length,
                   (select count(*) from chunks c where c.document_id = d.id) as chunk_count
            from documents d order by d.id desc
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(document_id: int) -> bool:
    with db() as conn:
        row = conn.execute("select id from documents where id = ?", (document_id,)).fetchone()
        if not row:
            return False
        conn.execute("delete from chunks where document_id = ?", (document_id,))
        conn.execute("delete from documents where id = ?", (document_id,))
    return True


def stats() -> dict[str, Any]:
    with db() as conn:
        doc_count = conn.execute("select count(*) from documents").fetchone()[0]
        chunk_count = conn.execute("select count(*) from chunks").fetchone()[0]
        qa_count = conn.execute("select count(*) from qa_history").fetchone()[0]
        avg_conf = conn.execute("select avg(confidence) from qa_history").fetchone()[0] or 0
    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "questions": qa_count,
        "average_confidence": round(avg_conf, 4),
    }


def clear_and_seed() -> None:
    with db() as conn:
        conn.execute("delete from qa_history")
        conn.execute("delete from chunks")
        conn.execute("delete from documents")
        add_document(conn, "人工智能方向课程综合实习讲义", DEFAULT_SAMPLE, source_type="sample")


def retrieve(question: str, top_k: int = 4, threshold: float = 0.02) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("select * from chunks").fetchall()
    if not rows:
        return []
    texts = [row["text"] for row in rows]
    vectors, idf = tfidf_vectors(texts)
    qv = vectorize_with_idf(question, idf)
    ranked: list[dict[str, Any]] = []
    for row, vec in zip(rows, vectors):
        chunk_keys = [k for k in (row["keywords"] or "").split(",") if k]
        # 关键词命中给一点加权，但分数主体仍是余弦相似度。
        boost = 0.05 if any(key and key in question for key in chunk_keys[:6]) else 0.0
        score = min(1.0, cosine(qv, vec) + boost)
        ranked.append(
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "document_title": row["document_title"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "keywords": chunk_keys,
                "score": round(score, 4),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    filtered = [item for item in ranked if item["score"] >= threshold][:top_k]
    return filtered or ranked[:top_k]


def all_document_text(limit_chars: int = 12000) -> str:
    with db() as conn:
        rows = conn.execute("select title, text from documents order by id desc").fetchall()
    parts: list[str] = []
    total = 0
    for row in rows:
        part = f"# {row['title']}\n{row['text']}"
        if total + len(part) > limit_chars:
            part = part[: max(0, limit_chars - total)]
        if part:
            parts.append(part)
            total += len(part)
        if total >= limit_chars:
            break
    return "\n\n".join(parts)


def save_history(question: str, answer: str, sources: list[dict[str, Any]], used_llm: bool, llm_error: Optional[str]) -> None:
    confidence = max([source["score"] for source in sources], default=0.0)
    with db() as conn:
        conn.execute(
            """
            insert into qa_history(question, answer, source_count, confidence, used_llm, llm_error, created_at)
            values(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question,
                answer,
                len(sources),
                confidence,
                1 if used_llm else 0,
                llm_error,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def history(limit: int = 30) -> list[dict[str, Any]]:
    with db() as conn:
        rows = conn.execute("select * from qa_history order by id desc limit ?", (limit,)).fetchall()
    return [dict(row) for row in rows]
