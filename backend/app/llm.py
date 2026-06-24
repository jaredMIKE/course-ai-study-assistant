from __future__ import annotations

import re
from typing import Any, Optional

import httpx

from . import config
from .nlp import compact_context, sentence_extract, keywords


def llm_available() -> bool:
    return bool(config.LLM_API_KEY)


async def call_llm(messages: list[dict[str, str]], max_tokens: int = 1200) -> tuple[Optional[str], bool, Optional[str]]:
    if not llm_available():
        return None, False, "未配置 LLM_API_KEY"
    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    try:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT) as client:
            response = await client.post(
                config.chat_completion_url(),
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip(), True, None
    except Exception as exc:
        return None, False, f"LLM 调用失败：{exc}"


def template_answer(question: str, sources: list[dict[str, Any]], mode: str = "student") -> str:
    facts: list[str] = []
    for source in sources:
        facts.extend(sentence_extract(source["text"], 2))
    facts = facts[:7]
    terms = keywords(question + " " + " ".join(source["text"] for source in sources), 8)
    prefix = "根据知识库检索结果，"
    if mode == "simple":
        prefix = "用更容易理解的话说，"
    elif mode == "exam":
        prefix = "按考试复习角度看，"
    elif mode == "strict":
        prefix = "严格依据资料可得，"

    if re.search(r"算法|关键|实现原理|原理", question):
        return f"{prefix}相关实现通常包括文本清洗、段落优先分块、关键词提取、TF-IDF 向量化、余弦相似度计算、Top-K 检索排序和基于检索上下文的答案生成。命中资料显示：{'；'.join(facts)}。"
    if re.search(r"rag|RAG|检索|向量|知识库|幻觉|来源|溯源", question):
        return (
            f"{prefix}RAG 的核心流程是资料导入、文本解析、清洗分块、向量化表示、相似度检索、"
            "提示词组装和答案生成。它把回答限定在课程资料上下文中，因此可以降低幻觉，并通过来源片段展示实现可追溯。"
        )
    if re.search(r"题|测验|测试|选择题|简答题|判断题|自测", question):
        return f"{prefix}可围绕 {', '.join(terms[:6])} 等知识点生成自测题，并要求学生结合来源片段回答。"
    if re.search(r"计划|复习|安排|怎么学|学习路线", question):
        return f"{prefix}建议先掌握 {', '.join(terms[:6])} 等高频知识点，再结合来源片段进行问答和自测巩固。"
    if facts:
        return f"{prefix}较相关的结论是：{'；'.join(facts)}。这些内容来自系统命中的课程片段，可在来源列表中查看依据。"
    return "知识库中暂未检索到足够相关的内容，建议先上传课程 PPT、PDF、Word 或粘贴讲义文本。"


async def rag_answer(question: str, sources: list[dict[str, Any]], mode: str = "student") -> tuple[str, bool, Optional[str]]:
    context = compact_context(sources)
    style = {
        "student": "回答要清晰，适合学生复习；必要时用分点说明。",
        "simple": "回答要通俗易懂，先讲结论，再解释概念。",
        "exam": "回答要贴近期末复习，突出重点、易错点和可能考法。",
        "strict": "回答要严格依据资料，不扩展资料外内容。",
    }.get(mode, "回答清晰、准确、基于材料。")
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个专业课程资料辅助学习系统中的 RAG 问答助手。"
                "必须优先依据给定的检索资料回答，不能凭空编造。"
                "若资料不足，要明确说明资料不足。回答中尽量引用[来源1]、[来源2]这样的依据编号。"
            ),
        },
        {
            "role": "user",
            "content": f"回答要求：{style}\n\n用户问题：{question}\n\n检索资料：\n{context}",
        },
    ]
    content, used, error = await call_llm(messages, max_tokens=1400)
    if content:
        return content, used, error
    return template_answer(question, sources, mode), False, error


async def llm_summary(text: str) -> tuple[Optional[str], bool, Optional[str]]:
    messages = [
        {"role": "system", "content": "你是课程复习资料整理助手。请只基于用户给出的资料生成摘要。"},
        {
            "role": "user",
            "content": (
                "请根据以下课程资料生成复习摘要，要求：\n"
                "1. 输出 6-8 条重点；\n2. 提炼核心概念；\n3. 给出复习建议；\n4. 不要编造资料外内容。\n\n"
                f"资料：\n{text[:9000]}"
            ),
        },
    ]
    return await call_llm(messages, max_tokens=1500)


async def llm_quiz(text: str) -> tuple[Optional[str], bool, Optional[str]]:
    messages = [
        {"role": "system", "content": "你是大学课程自测题生成助手。请只基于给定资料出题。"},
        {
            "role": "user",
            "content": (
                "请根据以下课程资料生成一组自测题，要求包含：2道判断题、2道简答题、1道应用题。"
                "每道题后给出参考答案和依据。不要编造资料外内容。\n\n"
                f"资料：\n{text[:9000]}"
            ),
        },
    ]
    return await call_llm(messages, max_tokens=1800)


async def llm_flashcards(text: str) -> tuple[Optional[str], bool, Optional[str]]:
    messages = [
        {"role": "system", "content": "你是课程复习卡片生成助手。请只基于给定资料生成记忆卡片。"},
        {
            "role": "user",
            "content": (
                "请根据以下课程资料生成 8 张复习卡片，格式为：正面：问题；背面：答案；依据：资料中的关键词或句子。"
                "卡片应覆盖核心概念、流程、算法和易错点。\n\n"
                f"资料：\n{text[:9000]}"
            ),
        },
    ]
    return await call_llm(messages, max_tokens=1800)


async def llm_review_plan(text: str) -> tuple[Optional[str], bool, Optional[str]]:
    messages = [
        {"role": "system", "content": "你是大学课程复习规划助手。请只基于给定资料制定复习计划。"},
        {
            "role": "user",
            "content": (
                "请根据以下课程资料生成一个三阶段复习计划，包含：基础梳理、重点突破、自测巩固。"
                "每阶段列出目标、任务和建议提问。不要写与资料无关的内容。\n\n"
                f"资料：\n{text[:9000]}"
            ),
        },
    ]
    return await call_llm(messages, max_tokens=1600)


def fallback_summary(text: str) -> dict[str, Any]:
    points = sentence_extract(text, 10)[:8]
    return {"points": points, "keywords": keywords(text, 14)}


def fallback_quiz(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    types = ["判断题", "简答题", "应用题", "概念解释题", "流程题"]
    items: list[dict[str, str]] = []
    for index, row in enumerate(chunks[:6]):
        key = (row.get("keywords") or ["核心概念"])[0] if isinstance(row.get("keywords"), list) else "核心概念"
        basis_list = sentence_extract(row.get("text", ""), 1)
        basis = basis_list[0] if basis_list else row.get("text", "")[:100]
        qtype = types[index % len(types)]
        if qtype == "判断题":
            question = f"判断：{key} 是当前资料中的关键知识点。请说明判断依据。"
        elif qtype == "应用题":
            question = f"如果要把 {key} 用到课程复习系统中，应如何设计功能？"
        elif qtype == "流程题":
            question = f"请结合资料说明 {key} 相关流程的主要步骤。"
        elif qtype == "概念解释题":
            question = f"请解释 {key} 的含义及其在资料中的作用。"
        else:
            question = f"请说明 {key} 为什么适合作为本资料的重点内容。"
        items.append({"type": qtype, "question": question, "answer": f"参考依据：{basis}", "basis": basis})
    return items


def fallback_flashcards(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for row in chunks[:8]:
        key = (row.get("keywords") or ["核心概念"])[0] if isinstance(row.get("keywords"), list) else "核心概念"
        basis_list = sentence_extract(row.get("text", ""), 1)
        basis = basis_list[0] if basis_list else row.get("text", "")[:120]
        cards.append({"front": f"{key} 是什么？", "back": basis, "source": row.get("document_title", "知识库片段")})
    return cards


def fallback_review_plan(text: str) -> dict[str, Any]:
    keys = keywords(text, 12)
    return {
        "keywords": keys,
        "stages": [
            {"name": "第一阶段：基础梳理", "task": f"通读资料，整理 {', '.join(keys[:4]) or '核心概念'} 等基础知识点。"},
            {"name": "第二阶段：重点突破", "task": f"围绕 {', '.join(keys[4:8]) or '重点模块'} 进行问答检索，查看来源片段并补充笔记。"},
            {"name": "第三阶段：自测巩固", "task": "使用自测题和复习卡片检查掌握情况，对答错内容重新检索溯源。"},
        ],
        "suggested_questions": [
            "这份资料的核心概念有哪些？",
            "哪些知识点最适合出简答题？",
            "请根据来源片段解释某个重点概念。",
            "请生成一组针对当前资料的自测题。",
        ],
    }
