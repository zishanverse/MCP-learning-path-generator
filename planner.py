"""
planner.py — Planning layer for the Learning Path Generator.

Responsibilities:
1. Call the LLM with a structured-JSON prompt.
2. Parse and validate the response against the LearningPath Pydantic schema.
3. Handle iterative continuation if fewer days are returned than requested.
4. Extract and validate YouTube video IDs.
5. Return a validated LearningPath object.

The LLM does NOT call any tools (YouTube, Drive, Notion) in this layer.
All external writes are delegated to actions.py.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from schemas import LearningPath, learning_path_to_markdown

load_dotenv()

# ---------------------------------------------------------------------------
# LLM initialisation
# ---------------------------------------------------------------------------

def _build_llm(model_name: str = "gemini-2.5-flash") -> Any:
    if "/" in model_name:
        # It's a Hugging Face model
        hf_token = os.getenv("HF_API_KEY", "").strip()
        if not hf_token:
            raise EnvironmentError("HF_API_KEY is not set in the environment.")
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=hf_token,
            temperature=0.7,
            max_new_tokens=2048,
            task="conversational",
        )
        return ChatHuggingFace(llm=llm)
    else:
        # It's a Google Gemini model
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = """\
You are a structured learning path generator.
You must return ONLY valid JSON — no markdown fences, no prose before or after.
The JSON must strictly match the schema below.

SCHEMA:
{
  "goal": "string — the user's goal restated concisely",
  "total_days": integer,
  "days": [
    {
      "day_number": integer,
      "topic": "string",
      "focus": "string — one sentence what the learner achieves today",
      "resources": [
        {
          "type": "Video | Course | Article | Documentation | Guide | Other",
          "title": "string",
          "url": "string — real, working URL (no placeholders)",
          "reason": "string — one sentence why it helps",
          "year": integer or null
        }
      ],
      "practice_task": "string",
      "learning_objectives": ["string", "string"]
    }
  ]
}

RULES:
- Every day must include at least ONE Video resource with a real youtube.com/watch?v= URL.
- Every day must include at least ONE non-Video resource.
- Never emit placeholder URLs (e.g. 'your-link-here', 'example.com').
- If you know a real working YouTube video ID, use: https://www.youtube.com/watch?v=VIDEO_ID
- If you DO NOT know a real working video ID for a topic, you MUST output a YouTube search URL instead: https://www.youtube.com/results?search_query=topic+name
- Prefer videos uploaded in the last 24 months. Include the upload year in the 'year' field.
- The 'total_days' field must match the exact number of objects in 'days'.
- Return ONLY the JSON object — nothing else.
"""


def _build_generation_prompt(user_goal: str, desired_days: int) -> str:
    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"User Goal: {user_goal}\n"
        f"Generate a learning path with exactly {desired_days} day(s)."
    )


def _build_continuation_prompt(user_goal: str, start_day: int, end_day: int) -> str:
    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"User Goal: {user_goal}\n"
        f"Continue the learning path. Return ONLY days {start_day} through {end_day}. "
        f"Set 'total_days' to {end_day - start_day + 1}. "
        f"Do NOT repeat earlier days. Return ONLY the JSON object."
    )


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[str]:
    """Strip markdown fences and return the JSON portion of an LLM response."""
    # Remove ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("`").strip()
    # Find the outermost { ... }
    start = text.find("{")
    if start == -1:
        return None
    # Walk to find matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _llm_response_to_text(response: Any) -> str:
    """Extract plain text from a LangChain LLM response object."""
    content = getattr(response, "content", None)
    if content is None:
        content = response
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Day count inference
# ---------------------------------------------------------------------------

def _infer_days(goal_text: str, fallback: int = 10) -> int:
    """Infer requested day count from user goal, clamped to [1, 60]."""
    text = goal_text.lower()
    for pattern, multiplier in [
        (r"(\d+)\s*(?:day|days)", 1),
        (r"(\d+)\s*(?:week|weeks)", 7),
        (r"(\d+)\s*(?:month|months)", 30),
    ]:
        m = re.search(pattern, text)
        if m:
            return max(1, min(60, int(m.group(1)) * multiplier))
    return max(1, min(60, fallback))


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate(
    user_goal: str,
    model_name: str = "gemini-2.5-flash",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> LearningPath:
    """Generate a validated LearningPath for the given user goal.

    Args:
        user_goal:         The user's free-text learning goal.
        model_name:        LLM model identifier.
        progress_callback: Optional callable for UI progress updates.

    Returns:
        A validated LearningPath Pydantic model.

    Raises:
        RuntimeError: On LLM failure or schema validation failure after retries.
    """
    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    llm = _build_llm(model_name)
    desired_days = _infer_days(user_goal)

    _progress(f"Planning {desired_days}-day learning path with {model_name}…")

    # --- Initial generation ---
    prompt = _build_generation_prompt(user_goal, desired_days)
    if isinstance(llm, BaseChatModel):
        response = llm.invoke([HumanMessage(content=prompt)])
    else:
        response = llm.invoke(prompt)
    raw_text = _llm_response_to_text(response)
    first_lp = _parse_learning_path(raw_text, user_goal, desired_days)

    if first_lp is None:
        raise RuntimeError(
            "LLM returned output that could not be parsed as a valid LearningPath. "
            f"Raw response (truncated): {raw_text[:500]}"
        )

    _progress(f"Received {len(first_lp.days)} day(s)…")

    # --- Iterative continuation if fewer days returned ---
    max_iterations = int(os.getenv("CONTINUATION_MAX_ITER", "4"))
    all_days = list(first_lp.days)
    iteration = 0

    while len(all_days) < desired_days and iteration < max_iterations:
        iteration += 1
        start_day = len(all_days) + 1
        end_day = desired_days
        _progress(f"Requesting continuation: Day {start_day}–{end_day} (attempt {iteration})…")

        cont_prompt = _build_continuation_prompt(user_goal, start_day, end_day)
        if isinstance(llm, BaseChatModel):
            cont_resp = llm.invoke([HumanMessage(content=cont_prompt)])
        else:
            cont_resp = llm.invoke(cont_prompt)
        cont_text = _llm_response_to_text(cont_resp)
        cont_lp = _parse_learning_path(cont_text, user_goal, end_day - start_day + 1)

        if cont_lp is None or not cont_lp.days:
            _progress("Continuation returned no parseable days — stopping.")
            break

        # Re-number continuation days to continue from where we left off
        for i, day in enumerate(cont_lp.days, start=start_day):
            day.day_number = i
        all_days.extend(cont_lp.days)
        _progress(f"Total days so far: {len(all_days)}")

    # --- Build final LearningPath ---
    final_lp = LearningPath(
        goal=first_lp.goal,
        total_days=len(all_days),
        days=all_days,
    )

    _progress("Learning path generation complete! ✅")
    return final_lp


def _parse_learning_path(
    raw_text: str,
    goal: str,
    expected_days: int,
) -> Optional[LearningPath]:
    """Try to parse raw LLM text into a LearningPath; returns None on failure."""
    json_str = _extract_json(raw_text)
    if not json_str:
        return None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # Inject goal if missing
    if not data.get("goal"):
        data["goal"] = goal
    if not data.get("total_days"):
        data["total_days"] = len(data.get("days", []))

    try:
        return LearningPath.model_validate(data)
    except ValidationError:
        # Try to salvage by relaxing constraints on individual days
        days = data.get("days", [])
        valid_days = []
        for d in days:
            try:
                from schemas import DayPlan
                valid_days.append(DayPlan.model_validate(d))
            except ValidationError:
                pass
        if not valid_days:
            return None
        data["days"] = [d.model_dump() for d in valid_days]
        data["total_days"] = len(valid_days)
        try:
            return LearningPath.model_validate(data)
        except ValidationError:
            return None
