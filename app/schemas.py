from typing import Literal, Optional

from pydantic import BaseModel, Field


class NewSessionResponse(BaseModel):
    session_id: str
    greeting: str
    returning: bool = False


class SessionInfoResponse(BaseModel):
    session_id: str
    stage: str
    active_intent: Optional[str]
    turn_count: int
    history_length: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    model: str
    retrieval_ready: bool = False
    index_chunks: int = 0
    embed_model: str = ""
    voice_ready: bool = False
    tools_enabled: bool = False
    voice_error: str = ""


# ---- WebSocket message contracts -------------------------------------------
# Client -> Server
class ClientChatMessage(BaseModel):
    type: Literal["message"] = "message"
    session_id: str
    text: str = Field(min_length=1, max_length=4000)


# Server -> Client (discriminated by "type")
class ServerStageEvent(BaseModel):
    type: Literal["stage"] = "stage"
    stage: str


class ServerTokenEvent(BaseModel):
    type: Literal["token"] = "token"
    text: str


class SourceRef(BaseModel):
    n: int
    title: str
    doc_id: str
    category: str
    score: float
    snippet: str


class RetrievalInfo(BaseModel):
    status: str  # ok | no_match | skipped | unavailable | disabled
    ms: float
    cached: bool


class ServerDoneEvent(BaseModel):
    """Assignment-1 fields are unchanged; `sources` and `retrieval` are additive."""
    type: Literal["done"] = "done"
    time_to_first_token: Optional[float]
    total_time: Optional[float]
    tokens_per_second: Optional[float]
    stage: str
    sources: list[SourceRef] = []
    retrieval: Optional[RetrievalInfo] = None
    tools: list[dict] = []
    timing: Optional[dict] = None


class ServerBlockedEvent(BaseModel):
    type: Literal["blocked"] = "blocked"
    text: str


class ServerErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
