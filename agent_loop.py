"""Single-agent skill runtime loop."""

import asyncio
import ast
import inspect
import json
import logging
import re
import uuid
from os import getenv
from typing import Callable, Literal

from httpx import ConnectTimeout, HTTPStatusError, ReadTimeout
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools.base import BaseTool
from langgraph.graph import END
from langgraph.types import Command

from .storage import (
    append_messages_ia,
    append_messages_tool,
    append_messages_user,
    is_complex_task_request,
    retrieve_last_input,
)
from .structs import NodeNames, ToolNames, WorkflowState
from .loader import list_skills
from .trace_logging import log_raw_event
from app_code.llm_protocol import use_native_llm_tools

calls_logger = logging.getLogger("calls")

# Number of consecutive action errors before returning control to the user.
MAX_CONSECUTIVE_ERRORS = 3
MAX_INVALID_TOOL_RETRIES = 3
JSON_PROTOCOL_MAX_PARSE_ATTEMPTS = 2
JSON_PROTOCOL_PARSE_RETRY_PROMPT = (
    "Your previous response could not be parsed as the required JSON protocol. "
    "Reply again with exactly one valid JSON object and no markdown, comments, or extra text. "
    "For a tool call use: "
    '{"type":"tool_call","name":"tool_name","args":{"arg_name":"value"}}. '
    "For a final answer use: "
    '{"type":"final","content":"user-facing answer"}.'
)
SINGLE_AGENT_SKILL_MODE_ENV = "USE_SINGLE_AGENT_SKILLS"
TASK_PROGRESS_BY_SESSION: dict[str, dict] = {}

SQL_ACTIONS = {
    ToolNames.VERIFY_SQL_SYNTAX.value,
    ToolNames.TEST_SQL_MODIFICATION_DRY_RUN.value,
    ToolNames.EXECUTE_SQL_MODIFICATION.value,
    ToolNames.EXECUTE_READ_QUERY.value,
    ToolNames.WRITE_SQL_FILE.value,
    ToolNames.READ_SQL_FILE.value,
    ToolNames.GET_TABLE_STRUCTURE.value,
}

CODE_ACTIONS = {
    ToolNames.WRITE_FILE.value,
    ToolNames.READ_FILE.value,
    ToolNames.EXECUTE_SCRIPT.value,
    ToolNames.RUN_TESTS.value,
    ToolNames.LINT_CODE.value,
    ToolNames.RUN_COMMAND.value,
    ToolNames.EXECUTE_CLI_COMMAND.value,
    ToolNames.GREP_SEARCH.value,
    ToolNames.MODIFY_CODE_SCRIPT.value,
}


def show_llm_content_in_status() -> bool:
    """Return whether LLM text content should be mirrored into browser status."""

    return getenv("SHOW_LLM_CONTENT_IN_STATUS", "false").lower() in {"1", "true", "yes", "on"}


def use_single_agent_skills() -> bool:
    """Return whether specialist skills should run inside the code node."""

    return getenv(SINGLE_AGENT_SKILL_MODE_ENV, "true").lower() in {"1", "true", "yes", "on"}


def _stringify_llm_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content).strip()


def _content_summary(content: str) -> str:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    if not first_line:
        return ""
    return first_line if len(first_line) <= 180 else f"{first_line[:177]}..."


def _status_skill_name(
    state: WorkflowState,
    action_name: str | None = None,
    skill_action_owners: dict[str, str] | None = None,
) -> str:
    if action_name == ToolNames.HANDOFF_TO_USER.value:
        return str(state.get("active_skill") or "system")
    if action_name and skill_action_owners and action_name in skill_action_owners:
        return skill_action_owners[action_name]
    return str(state.get("active_skill") or "system")


def _skill_step_label(skill_name: str) -> str:
    for skill in list_skills():
        if skill.name == skill_name:
            return str(skill.metadata.get("label") or skill.name.replace("-", " ").title())
    return skill_name.replace("-", " ").title()


async def _send_skill_status(
    state: WorkflowState,
    message: str,
    action_name: str | None = None,
    skill_action_owners: dict[str, str] | None = None,
    detail: dict | None = None,
) -> None:
    context = state.get("activity_context")
    if not context or not hasattr(context, "send_progress_event"):
        return
    skill_name = _status_skill_name(state, action_name, skill_action_owners)
    prefix = "[system]" if skill_name == "system" else f"[skill:{skill_name}]"
    await context.send_progress_event(
        f"{prefix} {message}",
        kind="agent_status",
        detail=detail,
    )


async def _send_llm_content_status(
    state: WorkflowState,
    content,
    action_name: str | None = None,
    skill_action_owners: dict[str, str] | None = None,
) -> None:
    if not show_llm_content_in_status():
        return
    text = _stringify_llm_content(content)
    if not text:
        return
    context = state.get("activity_context")
    if not context or not hasattr(context, "send_progress_event"):
        return
    skill_name = _status_skill_name(state, action_name, skill_action_owners)
    prefix = "[system]" if skill_name == "system" else f"[skill:{skill_name}]"
    await context.send_progress_event(
        f"{prefix} {_content_summary(text)}",
        kind="llm_content",
        detail={
            "type": "llm_content",
            "content": text,
            "parent_tool": action_name,
        },
    )


def _is_complex_task(state: WorkflowState) -> bool:
    return bool(state.get("task_plan_enabled"))


def _task_progress_lines(progress: dict) -> str:
    icons = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]", "error": "[!]"}
    return "\n".join(
        f"{icons.get(step['status'], '[ ]')} {step['label']}" for step in progress["steps"]
    )


async def _emit_task_progress(state: WorkflowState, skill_name: str, status: str) -> None:
    if not _is_complex_task(state):
        return
    session_key = str(state["session_id"])
    progress = TASK_PROGRESS_BY_SESSION.setdefault(
        session_key,
        {
            "steps": [
                {"skill": skill.name, "label": _skill_step_label(skill.name), "status": "pending"}
                for skill in list_skills()
            ]
        },
    )
    if not any(step["skill"] == skill_name for step in progress["steps"]):
        progress["steps"].append(
            {"skill": skill_name, "label": _skill_step_label(skill_name), "status": "pending"}
        )
    for step in progress["steps"]:
        if step["skill"] == skill_name:
            step["status"] = status
            break
    context = state.get("activity_context")
    if context and hasattr(context, "send_progress_event"):
        await context.send_progress_event(
            "[skill:plan] Task list updated\n" + _task_progress_lines(progress),
            kind="agent_status",
        )


def _sql_result_payload(tool_artifact: dict | None) -> dict | None:
    if not isinstance(tool_artifact, dict):
        return None
    context = tool_artifact.get("context")
    if not isinstance(context, str) or not context.strip():
        return None
    try:
        payload = json.loads(context)
    except Exception:
        return None
    rows = payload.get("rows")
    columns = payload.get("columns")
    if isinstance(rows, list) and isinstance(columns, list):
        return {
            "columns": columns,
            "rows": rows,
            "rows_returned": payload.get("rows_returned", len(rows)),
            "is_result_length_limited_by_tool": payload.get("is_result_length_limited_by_tool"),
        }
    return None


def _tool_detail(
    action_name: str,
    args: dict,
    result_content: str | None = None,
    tool_artifact: dict | None = None,
) -> dict | None:
    if action_name in SQL_ACTIONS:
        query = args.get("sql_query") or args.get("content")
        if not query and action_name == ToolNames.GET_TABLE_STRUCTURE.value:
            query = (
                f"database={args.get('database_name')}, "
                f"schema={args.get('schema_name')}, table={args.get('table_name')}"
            )
        detail = {
            "type": "sql",
            "tool": action_name,
            "query": query or "",
            "result": result_content or "",
            "csv_delimiter": getenv("SQL_RESULT_CSV_DELIMITER", ";"),
        }
        sql_result = _sql_result_payload(tool_artifact)
        if sql_result:
            detail["structured_result"] = sql_result
        return detail
    if action_name in CODE_ACTIONS:
        query = (
            args.get("content")
            or args.get("command")
            or args.get("command_parts")
            or args.get("file_path")
            or args.get("filename")
            or args.get("pattern")
            or ""
        )
        return {
            "type": "code",
            "tool": action_name,
            "label": action_name,
            "query": query,
            "result": result_content or "",
        }
    return None


def create_handoff_to_user_message(message: str) -> AIMessage:
    call_id = f"call_{uuid.uuid4()}"

    msg = AIMessage(
        content=message,
        tool_calls=[
            {
                "name": ToolNames.HANDOFF_TO_USER.value,
                "args": {
                    "user_prompt": message,
                },
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )

    msg.response_metadata = {
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "model_name": "system",
        "model": "system",
        "finish_reason": "system_handoff",
    }

    return msg


def _strip_json_code_fence(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    if raw.lower().startswith("json\n"):
        raw = raw[5:].strip()
    return raw


def _strip_json_comments(text: str) -> str:
    result: list[str] = []
    index = 0
    quote: str | None = None
    escaped = False
    while index < len(text):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if quote:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            result.append(char)
            index += 1
            continue
        if char == "/" and nxt == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and nxt == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index += 2
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _escape_control_chars_in_strings(text: str) -> str:
    result: list[str] = []
    quote: str | None = None
    escaped = False
    for char in text:
        if quote:
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char == quote:
                result.append(char)
                quote = None
                continue
            if char == "\n":
                result.append("\\n")
                continue
            if char == "\r":
                result.append("\\r")
                continue
            if char == "\t":
                result.append("\\t")
                continue
            result.append(char)
            continue
        if char in {'"', "'"}:
            quote = char
        result.append(char)
    return "".join(result)


def _replace_word_outside_strings(text: str, replacements: dict[str, str]) -> str:
    result: list[str] = []
    index = 0
    quote: str | None = None
    escaped = False
    word_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
    while index < len(text):
        char = text[index]
        if quote:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            result.append(char)
            index += 1
            continue
        replaced = False
        for old, new in replacements.items():
            end = index + len(old)
            if text[index:end] != old:
                continue
            before = text[index - 1] if index > 0 else ""
            after = text[end] if end < len(text) else ""
            if before in word_chars or after in word_chars:
                continue
            result.append(new)
            index = end
            replaced = True
            break
        if replaced:
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _single_quoted_strings_to_json(text: str) -> str:
    def convert(match: re.Match[str]) -> str:
        return json.dumps(match.group(1), ensure_ascii=False)

    return re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", convert, text)


def _quote_unquoted_keys(text: str) -> str:
    return re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_.-]*)(\s*:)',
        lambda match: f'{match.group(1)}"{match.group(2)}"{match.group(3)}',
        text,
    )


def _dirty_json_variants(text: str) -> list[str]:
    base = _strip_json_code_fence(text)
    base = base.strip().strip("\ufeff\u200b")
    if base.endswith(";"):
        base = base[:-1].strip()
    smart_quotes = str.maketrans({
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    })
    variants = [base, base.translate(smart_quotes)]
    cleaned_variants: list[str] = []
    for item in variants:
        item = _strip_json_comments(item)
        item = _escape_control_chars_in_strings(item)
        item = re.sub(r",\s*([}\]])", r"\1", item)
        cleaned_variants.append(item)
        cleaned_variants.append(_quote_unquoted_keys(item))
        single_fixed = _single_quoted_strings_to_json(item)
        cleaned_variants.append(single_fixed)
        cleaned_variants.append(_quote_unquoted_keys(single_fixed))
        cleaned_variants.append(
            _replace_word_outside_strings(
                _quote_unquoted_keys(single_fixed),
                {"True": "true", "False": "false", "None": "null"},
            )
        )

    seen: set[str] = set()
    result: list[str] = []
    for item in cleaned_variants:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _parse_jsonish_object(text: str) -> dict | None:
    for candidate in _dirty_json_variants(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        py_candidate = _replace_word_outside_strings(
            candidate,
            {"true": "True", "false": "False", "null": "None"},
        )
        try:
            parsed = ast.literal_eval(py_candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _json_object_candidates(text: str) -> list[str]:
    raw = _strip_json_code_fence(text)
    raw = raw.translate(str.maketrans({
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
    }))
    candidates: list[str] = [raw]
    stack: list[str] = []
    start: int | None = None
    quote: str | None = None
    escaped = False
    pairs = {"{": "}", "[": "]"}
    closing = set(pairs.values())
    for index, char in enumerate(raw):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'", "“", "”", "‘", "’"}:
            quote = '"' if char in {"“", "”"} else "'" if char in {"‘", "’"} else char
            continue
        if char in pairs:
            if not stack:
                start = index
            stack.append(pairs[char])
            continue
        if char in closing and stack and char == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                candidate = raw[start : index + 1]
                if candidate.startswith("{"):
                    candidates.append(candidate)
                start = None
    return candidates


def _extract_json_object(text: str) -> dict | None:
    raw = _strip_json_code_fence(text)
    if not raw:
        return None

    parsed_raw = _parse_jsonish_object(raw)
    if parsed_raw:
        return parsed_raw

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    objects: list[dict] = []
    for candidate in _json_object_candidates(raw):
        repaired = _parse_jsonish_object(candidate)
        if repaired:
            objects.append(repaired)
            continue
        index = 0
        while index < len(candidate):
            start = candidate.find("{", index)
            if start < 0:
                break
            try:
                parsed, offset = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                index = start + 1
                continue
            if isinstance(parsed, dict):
                objects.append(parsed)
            index = start + max(offset, 1)

    if not objects:
        return None

    for item in objects:
        response_type = str(item.get("type") or item.get("kind") or "").strip().lower()
        if response_type in {"tool_call", "tool", "action"} or "tool_calls" in item:
            return item

    for item in objects:
        if any(key in item for key in ("name", "tool", "action")) and any(
            key in item for key in ("args", "arguments")
        ):
            return item

    for item in objects:
        response_type = str(item.get("type") or item.get("kind") or "").strip().lower()
        if response_type in {"final", "answer", "message"} or "content" in item:
            return item

    return objects[-1]


def _response_metadata(result: AIMessage) -> dict:
    return getattr(result, "response_metadata", {}) or {
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "model_name": "unknown",
        "model": "unknown",
        "finish_reason": "json_protocol",
    }


def _infer_tool_name_from_arg_payload(
    payload: dict,
    available_actions: dict[str, BaseTool | Callable],
) -> str | None:
    """Infer a tool when a JSON-mode model returns only arguments."""

    payload_keys = {str(key) for key in payload}
    if {"table_name", "column_name"}.issubset(payload_keys) and "lineage_analyze_column" in available_actions:
        return "lineage_analyze_column"
    return None


def _looks_like_final_payload(payload: dict) -> bool:
    final_keys = {
        "target",
        "summary",
        "lineage_tree",
        "scripts_used",
        "cache",
        "cache_status",
        "ambiguities",
    }
    return bool(final_keys.intersection(str(key) for key in payload))


def _json_protocol_result_to_ai_message(
    result: AIMessage,
    available_actions: dict[str, BaseTool | Callable],
) -> AIMessage:
    """Convert JSON-mode LLM output into the runtime's normal AIMessage shape."""

    if use_native_llm_tools() or getattr(result, "tool_calls", None):
        return result

    payload = _extract_json_object(_stringify_llm_content(getattr(result, "content", "")))
    if not payload:
        logging.warning("LLM_TOOLS_MODE=json expected JSON, got non-JSON content.")
        return result

    response_type = str(payload.get("type") or payload.get("kind") or "").strip().lower()
    if response_type in {"final", "answer", "message"} or "content" in payload:
        content_value = payload.get("content") or payload.get("answer") or ""
        if isinstance(content_value, str):
            nested_payload = _parse_jsonish_object(content_value)
            if nested_payload and _looks_like_final_payload(nested_payload):
                content = json.dumps(nested_payload, ensure_ascii=False, indent=2)
            else:
                content = content_value
        else:
            content = json.dumps(content_value, ensure_ascii=False, indent=2)
        return AIMessage(
            id=getattr(result, "id", None),
            content=content,
            response_metadata=_response_metadata(result),
        )

    if _looks_like_final_payload(payload):
        return AIMessage(
            id=getattr(result, "id", None),
            content=json.dumps(payload, ensure_ascii=False, indent=2),
            response_metadata=_response_metadata(result),
        )

    raw_calls = payload.get("tool_calls")
    if isinstance(raw_calls, list) and raw_calls:
        first = raw_calls[0] if isinstance(raw_calls[0], dict) else {}
        tool_name = first.get("name") or first.get("tool") or first.get("action")
        args = first.get("args") or first.get("arguments") or {}
    else:
        tool_name = payload.get("name") or payload.get("tool") or payload.get("action")
        args = payload.get("args") or payload.get("arguments") or {}

    if not isinstance(tool_name, str) and not args:
        inferred_tool_name = _infer_tool_name_from_arg_payload(payload, available_actions)
        if inferred_tool_name:
            tool_name = inferred_tool_name
            args = dict(payload)
            logging.info(
                "Inferred JSON protocol tool_call from argument payload: %s",
                tool_name,
            )

    if not isinstance(tool_name, str) or tool_name not in available_actions:
        logging.warning(
            "LLM_TOOLS_MODE=json returned unknown tool '%s'. Available tools: %s",
            tool_name,
            sorted(available_actions.keys()),
        )
        return result
    if not isinstance(args, dict):
        args = {}

    logging.info("Converted JSON protocol output to tool_call: %s", tool_name)
    return AIMessage(
        id=getattr(result, "id", None),
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": args,
                "id": f"call_{uuid.uuid4()}",
                "type": "tool_call",
            }
        ],
        response_metadata=_response_metadata(result),
    )


def _json_protocol_parse_failed(original: AIMessage, converted: AIMessage) -> bool:
    if use_native_llm_tools() or getattr(original, "tool_calls", None):
        return False
    if getattr(converted, "tool_calls", None):
        return False
    if converted is not original:
        return False
    content = _stringify_llm_content(getattr(original, "content", ""))
    return bool(content.strip()) and _extract_json_object(content) is None


def _json_protocol_request_messages(messages: list) -> list:
    if use_native_llm_tools():
        return messages
    normalized = []
    for message in messages:
        if isinstance(message, dict):
            copy = dict(message)
            role = copy.get("role")
            tool_calls = copy.get("tool_calls") or copy.get("additional_kwargs", {}).get("tool_calls")
            if role == "assistant" and tool_calls:
                copy.pop("tool_calls", None)
                copy.pop("additional_kwargs", None)
                content = copy.get("content") or ""
                copy["content"] = (
                    f"{content}\n\nPrevious assistant tool call request: "
                    f"{json.dumps(tool_calls, ensure_ascii=False)}"
                ).strip()
                normalized.append(copy)
            elif role == "tool":
                content = copy.get("content") or ""
                tool_call_id = copy.get("tool_call_id")
                tool_label = f" for {tool_call_id}" if tool_call_id else ""
                normalized.append({
                    "role": "user",
                    "content": f"Previous tool result{tool_label}: {content}",
                })
            elif copy.get("content") is None:
                copy["content"] = ""
                normalized.append(copy)
            else:
                normalized.append(copy)
            continue
        tool_calls = getattr(message, "tool_calls", None) or getattr(
            message,
            "additional_kwargs",
            {},
        ).get("tool_calls")
        if isinstance(message, AIMessage) and tool_calls:
            content = _stringify_llm_content(getattr(message, "content", ""))
            normalized.append(AIMessage(
                content=(
                    f"{content}\n\nPrevious assistant tool call request: "
                    f"{json.dumps(tool_calls, ensure_ascii=False)}"
                ).strip(),
                id=getattr(message, "id", None),
                response_metadata=getattr(message, "response_metadata", {}),
            ))
            continue
        if isinstance(message, ToolMessage):
            content = _stringify_llm_content(getattr(message, "content", ""))
            tool_call_id = getattr(message, "tool_call_id", "")
            normalized.append(HumanMessage(
                content=(
                    f"Previous tool result"
                    f"{f' for {tool_call_id}' if tool_call_id else ''}: "
                    f"{content}"
                ),
                id=getattr(message, "id", None),
            ))
            continue
        if getattr(message, "content", "") is None:
            if hasattr(message, "model_copy"):
                normalized.append(message.model_copy(update={"content": ""}))
            else:
                message.content = ""
                normalized.append(message)
            continue
        normalized.append(message)
    return normalized


async def skip_remaining_tool_calls(
    current_agent: str,
    all_tool_calls: list,
    failed_tool_call_id: str,
    pool,
    session_id: str,
) -> None:
    """
    After a tool failure, declare all remaining tool_calls in the same batch
    as skipped. This is required to keep the message history consistent for
    chat completion providers that require one ToolMessage for every tool_call.
    """
    skip = False
    for tc in all_tool_calls:
        if tc["id"] == failed_tool_call_id:
            skip = True
            continue  # the failed one already has its ToolMessage
        if skip:
            skipped_msg = ToolMessage(
                content="",
                tool_call_id=tc["id"],
            )
            await append_messages_tool(
                current_agent,
                pool,
                session_id,
                skipped_msg,
                content_to_use=(
                    f"Tool '{tc['name']}' was not executed because "
                    "a previous tool in the same batch failed."
                ),
            )
            logging.info(
                "[agent_node] Skipped tool_call '%s' (id: %s) due to previous error.",
                tc["name"],
                tc["id"],
            )


async def agent_node(
    state: WorkflowState,
    llm_with_tools: BaseChatModel,
    skill_actions_by_name: dict[str, BaseTool | Callable],
    skill_action_owners: dict[str, str],
) -> Command[NodeNames]:
    """
    Runtime node function.
    Uses the active skill state history to decide the next action to perform.
    Executes skill actions, then updates message history and either continues or asks the user.
    """

    # 1. Extract the current agent's messages
    current_agent = state["next_agent"]

    # Logging for calls_logger
    if state["prev_agent"] != state["next_agent"]:
        calls_logger.info(
            f"> [skill:{_status_skill_name(state, skill_action_owners=skill_action_owners)}]"
        )
    prev_agent = current_agent

    logging.info(
        "--- LangGraph node=%s active_skill=%s session=%s ---",
        current_agent,
        state.get("active_skill"),
        state.get("session_id"),
    )

    try:
        current_agent_msg = state[current_agent]
        logging.debug(f"\nCURRENT_AGENT: {current_agent}")
        logging.info(
            "LangGraph message context: node=%s active_skill=%s messages=%d",
            current_agent,
            state.get("active_skill"),
            len(current_agent_msg["messages"]),
        )
        logging.debug(
            "LangGraph messages:\n%s",
            json.dumps(
                [m.model_dump() for m in current_agent_msg["messages"]],
                indent=5,
                ensure_ascii=False,
            ),
        )
        log_raw_event(
            "message_context",
            str(state.get("session_id")),
            node=str(current_agent),
            active_skill=state.get("active_skill"),
            messages=current_agent_msg["messages"],
        )
    except Exception as e:
        logging.error(f"Failed to retrieve messages for agent {current_agent}: {e}")
        return Command(
            update={"next_agent": NodeNames.USER_INPUT, "prev_agent": prev_agent},
            goto=NodeNames.USER_INPUT,
        )

    # LLM call with error handling (From ANAIA-94 branch)
    nb_attempts = 0
    MAX_LLM_ATTEMPTS = 3
    result = None
    fatal_error = False
    error_message = "Une erreur est survenue lors du traitement de votre demande."

    while nb_attempts < MAX_LLM_ATTEMPTS:
        nb_attempts += 1
        try:
            # Invoke the LLM with tools
            log_raw_event(
                "llm_request",
                str(state.get("session_id")),
                node=str(current_agent),
                active_skill=state.get("active_skill"),
                attempt=nb_attempts,
                messages=current_agent_msg["messages"],
                available_actions=sorted(skill_actions_by_name.keys()),
            )
            llm_messages = _json_protocol_request_messages(current_agent_msg["messages"])
            for json_parse_attempt in range(1, JSON_PROTOCOL_MAX_PARSE_ATTEMPTS + 1):
                raw_result = await llm_with_tools.ainvoke(llm_messages)
                converted_result = _json_protocol_result_to_ai_message(
                    raw_result,
                    skill_actions_by_name,
                )
                if not _json_protocol_parse_failed(raw_result, converted_result):
                    result = converted_result
                    break
                logging.warning(
                    "LLM_TOOLS_MODE=json parse failed for skill %s (%d/%d).",
                    _status_skill_name(state, skill_action_owners=skill_action_owners),
                    json_parse_attempt,
                    JSON_PROTOCOL_MAX_PARSE_ATTEMPTS,
                )
                log_raw_event(
                    "llm_json_parse_error",
                    str(state.get("session_id")),
                    node=str(current_agent),
                    active_skill=state.get("active_skill"),
                    attempt=nb_attempts,
                    json_parse_attempt=json_parse_attempt,
                    response=raw_result,
                )
                if json_parse_attempt == JSON_PROTOCOL_MAX_PARSE_ATTEMPTS:
                    result = converted_result
                    break
                llm_messages = _json_protocol_request_messages([
                    *current_agent_msg["messages"],
                    raw_result,
                    HumanMessage(content=JSON_PROTOCOL_PARSE_RETRY_PROMPT),
                ])
            await _send_llm_content_status(
                state,
                getattr(result, "content", ""),
                skill_action_owners=skill_action_owners,
            )
            log_raw_event(
                "llm_response",
                str(state.get("session_id")),
                node=str(current_agent),
                active_skill=state.get("active_skill"),
                attempt=nb_attempts,
                response=result,
                tool_calls=getattr(result, "tool_calls", []),
            )

            if hasattr(result, "tool_calls") and result.tool_calls:
                for tool_call in result.tool_calls:
                    logging.debug(
                        " ==> Tool: %s | id=%s | args=%s",
                        tool_call.get("name"),
                        tool_call.get("id"),
                        tool_call.get("args"),
                    )
            else:
                logging.debug("Aucun tool_call détecté à ce tour.")

            # if successful, break the retry loop
            break

        except HTTPStatusError as e:
            status_code = e.response.status_code
            log_raw_event(
                "llm_http_error",
                str(state.get("session_id")),
                node=str(current_agent),
                active_skill=state.get("active_skill"),
                status_code=status_code,
                response_text=getattr(e.response, "text", ""),
                attempt=nb_attempts,
            )

            # --- ERROR HANDLING RETRIEVABLE (408, 429, 500, 502, 503, 504) ---
            if status_code in [408, 429] or status_code >= 500:
                logging.warning(
                    "Retryable HTTP error %s for skill %s (%d/%d)",
                    status_code,
                    _status_skill_name(state, skill_action_owners=skill_action_owners),
                    nb_attempts,
                    MAX_LLM_ATTEMPTS,
                )
                if nb_attempts == MAX_LLM_ATTEMPTS:
                    logging.error(
                        "Max attempts reached for skill %s due to error %s",
                        _status_skill_name(state, skill_action_owners=skill_action_owners),
                        status_code,
                    )
                    error_message = (
                        "Le service momentanément indisponible. "
                        "Veuillez réessayer votre demande dans quelques instants."
                    )
                else:
                    await asyncio.sleep(1)
                continue

            # --- FATAL ERROR HANDLING 400 (Bad Request) ---
            elif status_code == 400:
                session_id = state.get("session_id", "N/A")
                logging.error(
                    "FATAL 400 for skill %s (node %s, session %s)",
                    _status_skill_name(state, skill_action_owners=skill_action_owners),
                    current_agent,
                    session_id,
                )
                error_message = (
                    f"Une erreur technique critique (400) est survenue. "
                    f"Référence de session : {session_id}.\n"
                    "Le service est indisponible pour cette conversation (probablement un contexte "
                    "trop long ou corrompu). Je vous invite à "
                    "**démarrer une nouvelle conversation**."
                )
                print(error_message)
                fatal_error = True
                break

            else:
                logging.error(
                    "HTTP error %s (%s) when calling LLM for skill %s. Response: %s",
                    e.response.status_code,
                    e.response.reason_phrase,
                    _status_skill_name(state, skill_action_owners=skill_action_owners),
                    e.response.text,
                )
                break

        # Timeout errors handling (Retryable)
        except (ReadTimeout, ConnectTimeout) as e:
            log_raw_event(
                "llm_timeout",
                str(state.get("session_id")),
                node=str(current_agent),
                active_skill=state.get("active_skill"),
                error=repr(e),
                attempt=nb_attempts,
            )
            logging.warning(
                "Timeout error when calling LLM for skill %s: %s.",
                _status_skill_name(state, skill_action_owners=skill_action_owners),
                str(e),
            )
            if nb_attempts == MAX_LLM_ATTEMPTS:
                logging.warning("Max attempts reached for skill %s due to timeout", _status_skill_name(state, skill_action_owners=skill_action_owners))
                error_message = (
                    "Le service momentanément indisponible. "
                    "Veuillez réessayer votre demande dans quelques instants."
                )
            else:
                await asyncio.sleep(1)

        # Any other unexpected errors
        except Exception as e:
            log_raw_event(
                "llm_unexpected_error",
                str(state.get("session_id")),
                node=str(current_agent),
                active_skill=state.get("active_skill"),
                error=repr(e),
                attempt=nb_attempts,
            )
            logging.error(
                "Unexpected error when calling LLM for skill %s: %s",
                _status_skill_name(state, skill_action_owners=skill_action_owners),
                str(e),
            )
            if nb_attempts == MAX_LLM_ATTEMPTS:
                error_message = (
                    "Le service momentanément indisponible. "
                    "Veuillez réessayer votre demande dans quelques instants."
                )
            await asyncio.sleep(1)

    if result is None:
        # On informe l'utilisateur
        handoff_msg = create_handoff_to_user_message(error_message or "Erreur interne.")
        await append_messages_ia(
            current_agent,
            state["pool"],
            state["session_id"],
            handoff_msg,
        )
        # Sauvegarder AUSSI dans user_input pour affichage
        await append_messages_user(
            NodeNames.USER_INPUT,
            state["pool"],
            state["session_id"],
            error_message or "Erreur interne.",
        )
        # Si erreur fatale -> END
        if fatal_error:
            return Command(
                update={"next_agent": END, "prev_agent": prev_agent},
                goto=END,
            )

        logging.error("ALL RETRIES FAILED for skill %s. Returning unavailable message", _status_skill_name(state, skill_action_owners=skill_action_owners))
        return Command(
            update={"next_agent": NodeNames.USER_INPUT, "prev_agent": prev_agent},
            goto=NodeNames.USER_INPUT,
        )

    # Managing tool_calls
    logging.info(
        "LLM call result for skill %s: tool_calls=%s",
        _status_skill_name(state, skill_action_owners=skill_action_owners),
        [call.get("name") for call in getattr(result, "tool_calls", [])],
    )

    if not hasattr(result, "tool_calls") or not result.tool_calls:
        if str(getattr(result, "content", "") or "").strip():
            final_content = str(result.content).strip()
            logging.info(
                "LLM produced final content for skill %s without an action call.",
                _status_skill_name(state, skill_action_owners=skill_action_owners),
            )
            await append_messages_ia(
                current_agent,
                state["pool"],
                state["session_id"],
                result,
            )
            await append_messages_user(
                NodeNames.USER_INPUT,
                state["pool"],
                state["session_id"],
                final_content,
            )
            return Command(
                update={
                    "next_agent": NodeNames.USER_INPUT,
                    "prev_agent": prev_agent,
                    "active_skill": None,
                    "caller_agent": NodeNames.USER_INPUT,
                },
                goto=NodeNames.USER_INPUT,
            )

        # Fallback mechanism if the agent refuses to call a tool
        logging.error("No action call produced for skill %s", _status_skill_name(state, skill_action_owners=skill_action_owners))

        handoff_msg = create_handoff_to_user_message(
            "Je n’ai pas pu poursuivre automatiquement. Pouvez-vous reformuler ?"
        )

        await append_messages_ia(
            current_agent,
            state["pool"],
            state["session_id"],
            handoff_msg,
        )
        # Save ToolMessage for each tool_call in handoff_msg to keep history consistent
        # with chat completion providers that require one tool result per tool call.
        for tc in handoff_msg.tool_calls:
            dummy_tool_msg = ToolMessage(content="", tool_call_id=tc["id"])
            await append_messages_tool(
                current_agent,
                state["pool"],
                state["session_id"],
                dummy_tool_msg,
                content_to_use="Handoff to user acknowledged.",
            )
        await append_messages_user(
            NodeNames.USER_INPUT,
            state["pool"],
            state["session_id"],
            handoff_msg.tool_calls[0]["args"]["user_prompt"],
        )

        return Command(
            update={"next_agent": NodeNames.USER_INPUT, "prev_agent": prev_agent},
            goto=NodeNames.USER_INPUT,
        )

    # Append AI message
    next_agent_to_call = current_agent
    await append_messages_ia(current_agent, state["pool"], state["session_id"], result)

    # Execute tools
    for tool_call in result.tool_calls:
        logging.info("Managing action call %s", tool_call)
        action_skill_name = _status_skill_name(
            state,
            tool_call["name"],
            skill_action_owners=skill_action_owners,
        )
        calls_logger.info(f"   > Skill: {action_skill_name}")
        calls_logger.info(f"   > Action: {tool_call['name']}")
        calls_logger.info(f"     Args: {tool_call['args']}")

        # --- Hallucination Check ---
        tool_name_str = tool_call["name"]
        try:
            if tool_name_str not in skill_actions_by_name:
                raise KeyError(f"Action {tool_name_str} is not available")
        except KeyError as e:
            logging.warning(f"Ignored hallucinated or unavailable tool: {tool_name_str} - {e}")
            available_tools = sorted(skill_actions_by_name.keys())
            retry_key = f"{current_agent}:{state.get('active_skill') or 'system'}"
            invalid_tool_retries = state.setdefault("invalid_tool_retries", {})
            invalid_tool_retries[retry_key] = invalid_tool_retries.get(retry_key, 0) + 1
            retry_count = invalid_tool_retries[retry_key]
            error_context = (
                f"Tool '{tool_name_str}' is not available in the current skill context. "
                f"Available tools are: {', '.join(available_tools)}. "
                "Use one of the available tools, or call load_skill to switch skills. "
                f"Invalid tool retry {retry_count}/{MAX_INVALID_TOOL_RETRIES}."
            )
            log_raw_event(
                "invalid_tool_call",
                str(state.get("session_id")),
                node=str(current_agent),
                active_skill=state.get("active_skill"),
                invalid_tool=tool_name_str,
                available_tools=available_tools,
                retry_count=retry_count,
                max_retries=MAX_INVALID_TOOL_RETRIES,
            )

            invalid_tool_msg = ToolMessage(content="", tool_call_id=tool_call["id"])
            await append_messages_tool(
                current_agent,
                state["pool"],
                state["session_id"],
                invalid_tool_msg,
                content_to_use=error_context,
            )
            await skip_remaining_tool_calls(
                current_agent,
                result.tool_calls,
                tool_call["id"],
                state["pool"],
                state["session_id"],
            )

            if retry_count < MAX_INVALID_TOOL_RETRIES:
                return Command(
                    update={
                        "next_agent": current_agent,
                        "prev_agent": prev_agent,
                        "active_skill": state.get("active_skill"),
                        "invalid_tool_retries": invalid_tool_retries,
                    },
                    goto=current_agent,
                )

            handoff_text = (
                "The model repeatedly tried to call an unavailable tool and could not continue. "
                f"Last invalid tool: '{tool_name_str}'. Available tools were: "
                f"{', '.join(available_tools)}."
            )
            handoff_msg = create_handoff_to_user_message(handoff_text)
            await append_messages_ia(current_agent, state["pool"], state["session_id"], handoff_msg)
            for tc in handoff_msg.tool_calls:
                dummy_tool_msg = ToolMessage(content="", tool_call_id=tc["id"])
                await append_messages_tool(
                    current_agent,
                    state["pool"],
                    state["session_id"],
                    dummy_tool_msg,
                    content_to_use="Handoff to user acknowledged.",
                )
            await append_messages_user(
                NodeNames.USER_INPUT,
                state["pool"],
                state["session_id"],
                handoff_text,
            )
            return Command(
                update={"next_agent": NodeNames.USER_INPUT, "prev_agent": prev_agent},
                goto=NodeNames.USER_INPUT,
            )

        tool_name = tool_name_str
        action_skill_name = _status_skill_name(
            state,
            tool_name,
            skill_action_owners=skill_action_owners,
        )
        logging.info(
            "LangGraph action start: skill=%s action=%s args=%s",
            action_skill_name,
            tool_name,
            tool_call.get("args", {}),
        )
        log_raw_event(
            "tool_call_start",
            str(state.get("session_id")),
            node=str(current_agent),
            active_skill=state.get("active_skill"),
            owner=action_skill_name,
            tool_name=tool_name,
            tool_call=tool_call,
        )
        await _emit_task_progress(state, action_skill_name, "in_progress")
        await _send_skill_status(
            state,
            f"Running `{tool_name}`",
            tool_name,
            skill_action_owners,
            _tool_detail(tool_name, tool_call.get("args", {})),
        )

        try:
            # Detect if tool is sync of async
            tool_obj = skill_actions_by_name[tool_name]
            actual_callable = getattr(tool_obj, "coroutine", None) or getattr(
                tool_obj, "func", None
            )
            # insert state in tool Args if demanded
            sig = inspect.signature(actual_callable)
            if "state" in sig.parameters:
                logging.info(f"[agent_node] Injecting state into {tool_name}")
                tool_call["args"]["state"] = state
            elif hasattr(actual_callable, "__globals__") and "_RUNTIME_STATE" in actual_callable.__globals__:
                logging.info("[agent_node] Injecting lightweight runtime state into %s globals", tool_name)
                actual_callable.__globals__["_RUNTIME_STATE"] = state
            # Invoke the tool
            try:
                res_tool_call: ToolMessage = await tool_obj.ainvoke(input=tool_call)
            finally:
                if hasattr(actual_callable, "__globals__") and "_RUNTIME_STATE" in actual_callable.__globals__:
                    actual_callable.__globals__["_RUNTIME_STATE"] = None
            tool_artifact = res_tool_call.artifact if isinstance(res_tool_call.artifact, dict) else {}
            llm_status_content = (
                tool_artifact.get("llm_status_content")
                or tool_artifact.get("llm_content")
            )
            if llm_status_content:
                await _send_llm_content_status(
                    state,
                    llm_status_content,
                    action_name=tool_name,
                    skill_action_owners=skill_action_owners,
                )
            logging.info(
                "LangGraph action finish: skill=%s action=%s result=%s",
                action_skill_name,
                tool_name,
                res_tool_call.content,
            )
            log_raw_event(
                "tool_call_finish",
                str(state.get("session_id")),
                node=str(current_agent),
                active_skill=state.get("active_skill"),
                owner=action_skill_name,
                tool_name=tool_name,
                result=res_tool_call,
            )
            await _emit_task_progress(state, action_skill_name, "done")
            await _send_skill_status(
                state,
                f"Finished `{tool_name}`",
                tool_name,
                skill_action_owners,
                _tool_detail(
                    tool_name,
                    tool_call.get("args", {}),
                    res_tool_call.content,
                    tool_artifact,
                ),
            )

            # reset error counter on successful tool call for the current agent
            if "error_counters" in state:
                state["error_counters"][current_agent] = 0

        except Exception as e:
            logging.exception(
                "LangGraph action failed: skill=%s action=%s",
                action_skill_name,
                tool_name,
            )
            log_raw_event(
                "tool_call_error",
                str(state.get("session_id")),
                node=str(current_agent),
                active_skill=state.get("active_skill"),
                owner=action_skill_name,
                tool_name=tool_name,
                tool_call=tool_call,
                error=repr(e),
            )
            await _emit_task_progress(state, action_skill_name, "error")

            # Increment error counter for the current agent
            if "error_counters" not in state:
                state["error_counters"] = {}
            state["error_counters"][current_agent] = (
                state["error_counters"].get(current_agent, 0) + 1
            )
            consecutive_errors = state["error_counters"][current_agent]
            logging.warning(
                "Skill %s has now %d/%d consecutive errors.",
                _status_skill_name(state, skill_action_owners=skill_action_owners),
                consecutive_errors,
                MAX_CONSECUTIVE_ERRORS,
            )

            # If max consecutive errors reached, return to the user-facing loop.
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                logging.error(
                    "Agent %s reached max consecutive errors. Forcing handoff to %s.",
                    current_agent,
                    state["caller_agent"],
                )
                state["error_counters"][current_agent] = 0  # reset counter after handoff
                caller = state["caller_agent"]

                handoff_msg = create_handoff_to_user_message(
                    f"The agent {current_agent} encountered "
                    f"{MAX_CONSECUTIVE_ERRORS} consecutive errors "
                    f"while executing the tool '{tool_name}' and cannot continue. "
                    f"Returning to {caller}.\n"
                    f"(Detail: {type(e).__name__}: {e})"
                )
                await append_messages_ia(
                    current_agent, state["pool"], state["session_id"], handoff_msg
                )

                if caller == NodeNames.USER_INPUT:
                    await append_messages_user(
                        NodeNames.USER_INPUT,
                        state["pool"],
                        state["session_id"],
                        f"The agent {current_agent} encountered "
                        f"{MAX_CONSECUTIVE_ERRORS} consecutive errors "
                        f"and cannot continue. Please rephrase your request.",
                    )
                else:
                    await append_messages_user(
                        caller,
                        state["pool"],
                        state["session_id"],
                        f"[SYSTEM] The agent {current_agent} has failed "
                        f"{MAX_CONSECUTIVE_ERRORS} times in a row. Task interrupted.",
                    )

                # Declare remaining tool_calls in the batch as skipped
                await skip_remaining_tool_calls(
                    current_agent,
                    result.tool_calls,
                    tool_call["id"],
                    state["pool"],
                    state["session_id"],
                )

                return Command(
                    update={"next_agent": caller, "prev_agent": prev_agent},
                    goto=caller,
                )

            # Isolated error : We store the error in memory and let the agent
            # try again with a different tool on the next turn
            error_tool_msg = ToolMessage(
                content="",
                tool_call_id=tool_call["id"],
            )
            await append_messages_tool(
                current_agent,
                state["pool"],
                state["session_id"],
                error_tool_msg,
                content_to_use=(
                    f"An error occurred while executing tool '{tool_name}': "
                    f"{type(e).__name__}: {e}\n"
                    "The agent will try to continue and find an alternative solution."
                ),
            )
            # Declare remaining tool_calls in the batch as skipped
            await skip_remaining_tool_calls(
                current_agent,
                result.tool_calls,
                tool_call["id"],
                state["pool"],
                state["session_id"],
            )
            return Command(
                update={"next_agent": current_agent},
                goto=current_agent,
            )

        # Handle artifacts logic
        if res_tool_call.artifact is None:
            res_tool_call.artifact = {}

        if "next_agent" not in res_tool_call.artifact:
            res_tool_call.artifact["next_agent"] = current_agent

        logging.info(
            "LangGraph action artifact: action=%s next=%s active_skill=%s content_chars=%d",
            tool_name,
            res_tool_call.artifact.get("next_agent"),
            res_tool_call.artifact.get("active_skill", action_skill_name),
            len(str(res_tool_call.content or "")),
        )

        # Logic selection based on next agent
        proposed_next_agent = res_tool_call.artifact["next_agent"]
        current_active_skill = state.get("active_skill")
        active_skill_to_set = res_tool_call.artifact.get("active_skill")
        if not active_skill_to_set:
            if action_skill_name == "system":
                active_skill_to_set = current_active_skill
            elif current_active_skill and action_skill_name != current_active_skill:
                active_skill_to_set = current_active_skill
            else:
                active_skill_to_set = action_skill_name
        if (
            use_single_agent_skills()
            and current_agent == NodeNames.CODE_AGENT
            and proposed_next_agent != NodeNames.CODE_AGENT
            and proposed_next_agent not in {NodeNames.USER_INPUT, END}
        ):
            res_tool_call.artifact["active_skill"] = active_skill_to_set
            res_tool_call.artifact["next_agent"] = NodeNames.CODE_AGENT
            proposed_next_agent = NodeNames.CODE_AGENT

        # Case 1: Agent continues working on itself
        if proposed_next_agent == current_agent:
            await append_messages_tool(
                current_agent,
                state["pool"],
                state["session_id"],
                res_tool_call,
                content_to_use=res_tool_call.artifact.get("context", res_tool_call.content),
            )

        # Case 2: Handoff to USER or END
        elif proposed_next_agent == NodeNames.USER_INPUT or proposed_next_agent == END:
            next_agent_to_call = proposed_next_agent

            await append_messages_tool(
                current_agent,
                state["pool"],
                state["session_id"],
                res_tool_call,
                content_to_use=res_tool_call.content,
            )

            # If USER_INPUT, explicitly save user message context if needed
            if proposed_next_agent == NodeNames.USER_INPUT:
                logging.debug("Final handoff to USER_INPUT confirmed.")
                # We use the artifact context as the message to show/store for the user transition
                await append_messages_user(
                    NodeNames.USER_INPUT,
                    state["pool"],
                    state["session_id"],
                    res_tool_call.artifact.get("context", ""),
                )
            break

        # Case 3: Internal skill context transition.
        else:
            if next_agent_to_call != current_agent and next_agent_to_call != proposed_next_agent:
                logging.warning("Multiple handoffs detected. Priority to the last one.")

            next_agent_to_call = proposed_next_agent

            await append_messages_tool(
                current_agent,
                state["pool"],
                state["session_id"],
                res_tool_call,
                content_to_use=res_tool_call.content,
            )

            # Propagate context to the next agent
            await append_messages_user(
                next_agent_to_call,
                state["pool"],
                state["session_id"],
                f"[Agent {current_agent}]: {res_tool_call.artifact.get('context', '')}",
            )

    logging.debug("State after tools: %s", state)
    logging.info("Handing control from %s to %s", current_agent, next_agent_to_call)

    update_dict = {"next_agent": next_agent_to_call, "prev_agent": prev_agent}
    if use_single_agent_skills() and next_agent_to_call == NodeNames.USER_INPUT:
        update_dict["active_skill"] = None
    elif "active_skill_to_set" in locals() and active_skill_to_set:
        update_dict["active_skill"] = active_skill_to_set
    if (
        use_single_agent_skills()
        and "res_tool_call" in locals()
        and isinstance(getattr(res_tool_call, "artifact", None), dict)
        and res_tool_call.artifact.get("caller_agent")
    ):
        update_dict["caller_agent"] = res_tool_call.artifact["caller_agent"]
    if current_agent != next_agent_to_call:
        update_dict["caller_agent"] = current_agent

    return Command(
        update=update_dict,
        goto=next_agent_to_call,
    )


async def user_node(
    state: WorkflowState,
) -> Command[Literal[NodeNames.CODE_AGENT, END]]:  # type: ignore
    """
    Specific user node: read input from the user.
    Can either return to caller agent, or go to the graph's end.
    """
    last_input = await retrieve_last_input(state["pool"], state["session_id"], NodeNames.USER_INPUT)

    if last_input:
        print("\n--------------------------\n")
        user_prompt = last_input + "\n(Enter '0' or 'exit' to exit)\n"
    else:
        user_prompt = "Hi! How can I help you today?\n(Enter '0' or 'exit' to exit)\n"

    try:
        user_input = input("AgenticAI: " + user_prompt + "You: ")
    except EOFError:
        logging.info("EOF received — ending session")
        return Command(goto=END)
    logging.info("Received user input: %s", user_input)

    user_input_clean = user_input.strip().lower()
    if user_input_clean in {"0", "exit"}:
        logging.info("Ending session")
        return Command(goto=END)

    caller_agent = state["caller_agent"]

    # --- Feedback Analysis (from colleague's code) ---
    # TODO what's this, is this necessary?
    feedback_type = "FEEDBACK"
    full_message = user_input

    if any(
        word in user_input_clean
        for word in ["oui", "ok", "yes", "valid", "envoie", "go", "confirm"]
    ):
        feedback_type = "CONFIRMED"
        full_message = f"User confirmed: {user_input} - Proceed."
    elif any(word in user_input_clean for word in ["non", "no", "annule", "stop", "cancel"]):
        feedback_type = "CANCELLED"
        full_message = f"User cancelled: {user_input}"
    else:
        full_message = f"User feedback: {user_input}"

    # Log/Tag for the agent
    full_message_with_tag = f"[{feedback_type}] {full_message}"
    # -------------------------------------------------

    await append_messages_user(
        caller_agent, state["pool"], state["session_id"], full_message_with_tag
    )

    return Command(
        update={
            "next_agent": caller_agent,
            "caller_agent": NodeNames.USER_INPUT,
            "task_plan_enabled": is_complex_task_request(user_input),
        },
        goto=caller_agent,
    )
