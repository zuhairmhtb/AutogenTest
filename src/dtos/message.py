from dataclasses import dataclass

@dataclass
class Message:
    query: str

@dataclass
class QueryMessage(Message):
    context: str