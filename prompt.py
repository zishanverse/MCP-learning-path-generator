"""
prompt.py — Prompt constants for the Learning Path Generator.

The primary prompt (STRUCTURED_JSON_PROMPT) instructs the LLM to return
validated JSON matching the LearningPath schema defined in schemas.py.

The legacy user_goal_prompt is kept for backward compatibility with the
old utils.py/run_agent_sync path but should not be used in new code.
"""

# ---------------------------------------------------------------------------
# Primary prompt — used by planner.py
# ---------------------------------------------------------------------------

STRUCTURED_JSON_PROMPT = """\
You are a structured learning path generator.
You must return ONLY valid JSON — no markdown fences, no prose before or after.

RULES:
- Every day must have at least ONE Video resource (canonical YouTube URL: https://www.youtube.com/watch?v=VIDEO_ID).
- Every day must have at least ONE non-Video resource (course, article, documentation, etc.).
- Never emit placeholder URLs. All URLs must be real and working.
- Prefer videos uploaded within the last 24 months. Set the 'year' field to the upload year.
- 'total_days' must equal the number of objects in 'days'.
- Return ONLY the JSON object — nothing else.
"""

# ---------------------------------------------------------------------------
# Legacy prompt — kept for backward compatibility only
# Do NOT use this in new code. Use planner.py instead.
# ---------------------------------------------------------------------------

user_goal_prompt = """\
[LEGACY] Main Instruction: You are a day wise learning path generator. You will be given a goal.
Produce a comprehensive day-wise learning path, capture it in a Drive document/Notion page when
those tools are available, and mirror the core video content in a YouTube playlist.

Note: This prompt is retained for backward compatibility. New code should use planner.py
and the structured JSON output path instead.
"""
