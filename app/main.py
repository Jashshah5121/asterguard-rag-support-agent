from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent.controller import AgentController
from app.agent.orchestrator import AgentOrchestrator
from app.agent.responder import AgentResponder

from app.config import settings

from app.context.memory import ContextMemory
from app.context.resolver import ContextResolver

from app.models.session import (
    ConversationTurn,
    SessionState,
)

from app.orders.repository import OrderRepository
from app.orders.service import OrderService

from app.rag.conflicts import ConflictDetector
from app.rag.evidence import EvidenceSelector
from app.rag.index import VectorIndex
from app.rag.lexical import LexicalRetriever
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import HybridRetriever
from app.rag.store import ChunkStore

from app.tools.order_lookup import OrderLookupTool


app = FastAPI(
    title="Aster & Row Support Agent",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    blocked: bool = False
    handoff: bool = False
    sources: list[str] = []


sessions: Dict[str, SessionState] = {}

orchestrator: AgentOrchestrator | None = None
responder: AgentResponder | None = None

context_memory = ContextMemory()
context_resolver = ContextResolver()


def build_application():

    chunk_store = ChunkStore()

    chunks = chunk_store.load(
        settings.index_path
    )

    vector_index = VectorIndex()

    vector_index.load(
        settings.index_path
    )

    lexical_retriever = LexicalRetriever()

    lexical_retriever.build(
        chunks
    )

    hybrid_retriever = HybridRetriever(
        vector_index=vector_index,
        lexical_retriever=lexical_retriever,
    )

    evidence_selector = EvidenceSelector()

    conflict_detector = ConflictDetector()

    rag_pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        evidence_selector=evidence_selector,
        conflict_detector=conflict_detector,
    )

    repository = OrderRepository(
        settings.orders_path
    )

    order_service = OrderService(
        repository
    )

    order_tool = OrderLookupTool(
        order_service
    )

    controller = AgentController()

    agent_orchestrator = AgentOrchestrator(
        controller=controller,
        rag_pipeline=rag_pipeline,
        order_tool=order_tool,
        chunks=chunks,
    )

    agent_responder = AgentResponder()

    return (
        agent_orchestrator,
        agent_responder,
    )


@app.on_event("startup")
def startup():
    global orchestrator
    global responder

    orchestrator, responder = build_application()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "aster-row-support-agent",
    }


@app.delete("/sessions/{session_id}")
def clear_session(
    session_id: str,
):
    session_id = session_id.strip()

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="session_id is required",
        )

    sessions.pop(
        session_id,
        None,
    )

    return {
        "status": "cleared",
        "session_id": session_id,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):
    if not request.session_id.strip():
        raise HTTPException(
            status_code=400,
            detail="session_id is required",
        )

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="message is required",
        )

    if (
        orchestrator is None
        or responder is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Agent is not initialized",
        )

    session_id = request.session_id.strip()

    original_query = request.message.strip()

    session = sessions.get(
        session_id
    )

    if session is None:
        session = SessionState(
            session_id=session_id
        )
        sessions[session_id] = session

    original_query = request.message.strip()

    resolver_query = context_resolver.resolve(
        query=original_query,
        session=session,
    )

    is_followup = context_memory.is_followup(
        original_query
    )

    if is_followup:
        contextual_query = context_memory.resolve_followup(
            query=original_query,
            session=session,
        )

        agent_query = contextual_query
        retrieval_query = contextual_query
    else:
        agent_query = original_query
        retrieval_query = resolver_query

    result = orchestrator.run(
        query=agent_query,
        session=session,
        retrieval_query=retrieval_query,
    )

    generated = responder.generate(
        query=agent_query,
        session=session,
        result=result,
    )

    session.recent_turns.append(
        ConversationTurn(
            role="user",
            content=original_query,
        )
    )

    session.recent_turns.append(
        ConversationTurn(
            role="assistant",
            content=generated.answer,
        )
    )

    session.recent_turns = (
        session.recent_turns[-10:]
    )

    context_memory.update(
        query=original_query,
        session=session,
        answer=generated.answer,
    )

    if (
        result.decision.order_id
        and result.order_data is not None
    ):
        session.active_order_id = (
            result.decision.order_id
        )

    source_text = " ".join(
        generated.sources
    ).lower()

    answer_text = generated.answer.lower()

    if (
        "breeze tumbler" in source_text
        or "breeze tumbler" in answer_text
    ):
        session.entities[
            "product"
        ] = "Breeze Tumbler"

    elif (
        "atlas weekender" in source_text
        or "atlas weekender" in answer_text
    ):
        session.entities[
            "product"
        ] = "Atlas Weekender"

    return ChatResponse(
        session_id=session_id,
        answer=generated.answer,
        blocked=result.blocked,
        handoff=generated.handoff,
        sources=generated.sources,
    )


app.mount(
    "/",
    StaticFiles(
        directory="frontend",
        html=True,
    ),
    name="frontend",
)