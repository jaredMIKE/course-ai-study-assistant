from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class DocumentIn(BaseModel):
    title: str = Field(default="粘贴课程资料", max_length=120)
    text: str = Field(min_length=1)


class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)
    threshold: float = Field(default=0.02, ge=0, le=1)
    mode: str = Field(default="student")
    use_llm: bool = True


class ApiResponse(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
