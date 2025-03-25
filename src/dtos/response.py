from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Response:
    content: str
    status_code: int
    error: Optional[str] = None

@dataclass
class QueryResponse(Response):
    id: Optional[str] = None
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None  # in seconds
    model_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    context_length: Optional[int] = None
    query_length: Optional[int] = None