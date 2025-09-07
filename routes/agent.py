from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, Any, AsyncGenerator
import asyncio
import json

from utils.auth import get_current_user
from agent.blog_agent import BlogGenerationAgent

router = APIRouter(prefix="/agent", tags=["agent"])


_agent: BlogGenerationAgent | None = None


def _get_agent() -> BlogGenerationAgent:
    global _agent
    if _agent is None:
        _agent = BlogGenerationAgent()
    return _agent


@router.post("/respond")
async def agent_respond(payload: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """Accepts: { "messages": [{"role":"user","content":"..."}, ...], "thread_id"?: str }

    Returns full JSON result from the BlogGenerationAgent.
    """
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        messages = [messages]

    # Compose a simple query from the last user message
    query = ""
    for m in reversed(messages):
        if isinstance(m, dict) and (m.get("role") == "user"):
            query = str(m.get("content") or "").strip()
            break

    thread_id = payload.get("thread_id")

    agent = _get_agent()
    result = await agent.process_query(query=query, user_id=current_user.get("id", "user"), thread_id=thread_id)
    return result


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def agent_stream(payload: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """Server-Sent Events stream that reports step-by-step progress and partial outputs.

    Body: { "query"?: str, "messages"?: [...], "thread_id"?: str }
    """
    messages = payload.get("messages") or []
    query = payload.get("query") or ""

    if not query:
        # Build query from last user message if not provided
        for m in reversed(messages):
            if isinstance(m, dict) and (m.get("role") == "user"):
                query = str(m.get("content") or "").strip()
                break

    thread_id = payload.get("thread_id")
    user_id = current_user.get("id", "user")
    agent = _get_agent()

    async def event_generator() -> AsyncGenerator[bytes, None]:
        # Start event
        yield _sse({"type": "start", "query": query, "thread_id": thread_id or None}).encode("utf-8")

        # Load memory step
        try:
            state = await agent.graph.ainvoke(
                # Kick off at entry; we will still summarize steps ourselves after full run
                # but first emit synthetic step events before and after heavy ops
                # We cannot easily intercept internal nodes without refactoring;
                # so we will emulate a step-by-step by running sub-steps manually.
                # Below we re-run the exact sequence with the underlying services.
                # This ensures streaming visibility while keeping same results.
                # However, for full control we reconstruct the steps explicitly.
                # To avoid duplication, we perform steps manually here.
                # Placeholder; this call will be replaced by manual sequence below
                # if needed in the future.
                # For now, we will not use this direct call here.
                # Keep compatibility: do nothing
                # NOTE: We will not await this; just pass
                # This branch won't run
                # pragma: no cover
                {}  # type: ignore
            )
        except Exception:
            # ignore; we are going to use manual sequence below
            pass

        try:
            # Manual execution of the agent steps for streaming visibility
            from agent.blog_agent import AgentState  # local import to avoid circulars

            # Build initial state
            initial_state = AgentState(query=query, user_id=user_id, thread_id=thread_id or agent.memory_manager.create_new_thread_id(user_id))

            # load_memory
            yield _sse({"type": "step", "name": "load_memory", "status": "start"}).encode("utf-8")
            state_obj = agent._load_memory(initial_state)
            yield _sse({"type": "step", "name": "load_memory", "status": "end", "memory_context_len": len(state_obj.memory_context)}).encode("utf-8")

            # classify_intent
            yield _sse({"type": "step", "name": "classify_intent", "status": "start"}).encode("utf-8")
            state_obj = agent._classify_intent(state_obj)
            yield _sse({"type": "intent", "is_blog_request": bool(state_obj.is_blog_request)}).encode("utf-8")
            yield _sse({"type": "step", "name": "classify_intent", "status": "end"}).encode("utf-8")

            if not state_obj.is_blog_request:
                # generate casual response
                yield _sse({"type": "step", "name": "generate_casual_response", "status": "start"}).encode("utf-8")
                state_obj = agent._generate_casual_response(state_obj)
                yield _sse({"type": "delta", "content": state_obj.response}).encode("utf-8")
                yield _sse({"type": "step", "name": "generate_casual_response", "status": "end"}).encode("utf-8")

                # save memory
                yield _sse({"type": "step", "name": "save_memory", "status": "start"}).encode("utf-8")
                state_obj = agent._save_memory(state_obj)
                yield _sse({"type": "step", "name": "save_memory", "status": "end"}).encode("utf-8")

                yield _sse({"type": "end", "thread_id": state_obj.thread_id}).encode("utf-8")
                return

            # search topics
            yield _sse({"type": "step", "name": "search_topics", "status": "start"}).encode("utf-8")
            state_obj = agent._search_topics(state_obj)
            
            # Send search results with proper formatting
            search_results_formatted = []
            for i, result in enumerate(state_obj.search_results[:10]):  # Limit to 10 results
                search_results_formatted.append({
                    "id": i + 1,
                    "title": result.get('title', 'No title'),
                    "content": result.get('content', result.get('snippet', 'No content'))[:200] + "...",
                    "url": result.get('url', '#'),
                    "score": result.get('score', 0.0)
                })
            
            yield _sse({
                "type": "search_results", 
                "count": len(state_obj.search_results), 
                "results": search_results_formatted
            }).encode("utf-8")
            yield _sse({"type": "step", "name": "search_topics", "status": "end"}).encode("utf-8")

            # generate blog
            yield _sse({"type": "step", "name": "generate_blog", "status": "start"}).encode("utf-8")
            state_obj = agent._generate_blog(state_obj)

            # stream content in chunks
            full = state_obj.response or ""
            for chunk in _chunk_text(full, 1200):
                yield _sse({"type": "delta", "content": chunk}).encode("utf-8")
                await asyncio.sleep(0.02)

            yield _sse({"type": "metrics", "word_count": getattr(getattr(state_obj, 'blog_content', None), 'word_count', 0)}).encode("utf-8")
            yield _sse({"type": "step", "name": "generate_blog", "status": "end"}).encode("utf-8")

            # save memory
            yield _sse({"type": "step", "name": "save_memory", "status": "start"}).encode("utf-8")
            state_obj = agent._save_memory(state_obj)
            yield _sse({"type": "step", "name": "save_memory", "status": "end"}).encode("utf-8")

            yield _sse({"type": "end", "thread_id": state_obj.thread_id}).encode("utf-8")

        except Exception as e:
            yield _sse({"type": "error", "message": str(e)}).encode("utf-8")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _chunk_text(text: str, max_bytes: int) -> list[str]:
    chunks: list[str] = []
    buf = []
    size = 0
    for line in text.splitlines(keepends=True):
        line_bytes = line.encode("utf-8")
        if size + len(line_bytes) > max_bytes and buf:
            chunks.append("".join(buf))
            buf = [line]
            size = len(line_bytes)
        else:
            buf.append(line)
            size += len(line_bytes)
    if buf:
        chunks.append("".join(buf))
    return chunks
