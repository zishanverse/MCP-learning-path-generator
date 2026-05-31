"""
schemas.py — Pydantic data models for the Learning Path Generator.

The LLM is instructed to return a JSON object matching the LearningPath schema.
After validation, the app converts the model to markdown for display/export.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Resource(BaseModel):
    """A single learning resource (video, course, article, etc.)."""

    type: Literal["Video", "Course", "Article", "Documentation", "Guide", "Other"] = Field(
        description="The category of the resource."
    )
    title: str = Field(description="Full title of the resource.")
    url: str = Field(description="Canonical, working URL to the resource.")
    reason: str = Field(description="One sentence explaining why this resource helps.")
    year: Optional[int] = Field(
        default=None,
        description="Publication or upload year (e.g. 2024). Optional but preferred for videos.",
    )

    @field_validator("url")
    @classmethod
    def url_must_not_be_placeholder(cls, v: str) -> str:
        blocked = {"placeholder", "INSERT_", "your-link", "example.com", "LINK_HERE"}
        for bad in blocked:
            if bad.lower() in v.lower():
                raise ValueError(f"URL appears to be a placeholder: {v!r}")
        return v

    @field_validator("title")
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Resource title must not be empty.")
        return v.strip()


class DayPlan(BaseModel):
    """A single day in the learning path."""

    day_number: int = Field(ge=1, description="The day index (1-based).")
    topic: str = Field(description="Short topic name for this day.")
    focus: str = Field(description="One-sentence summary of what the learner will accomplish.")
    resources: List[Resource] = Field(
        min_length=2,
        description="At least one video and one non-video resource.",
    )
    practice_task: str = Field(description="A short, actionable hands-on task for the day.")
    learning_objectives: List[str] = Field(
        min_length=2,
        max_length=5,
        description="2-5 measurable outcomes the learner should achieve.",
    )

    @model_validator(mode="after")
    def must_have_video_and_non_video(self) -> "DayPlan":
        types = [r.type for r in self.resources]
        if "Video" not in types:
            raise ValueError(f"Day {self.day_number} must include at least one Video resource.")
        non_video = [t for t in types if t != "Video"]
        if not non_video:
            raise ValueError(
                f"Day {self.day_number} must include at least one non-Video resource."
            )
        return self


class LearningPath(BaseModel):
    """The complete structured learning path returned by the LLM."""

    goal: str = Field(description="The original user goal, restated concisely.")
    total_days: int = Field(ge=1, le=60, description="Total number of days planned.")
    days: List[DayPlan] = Field(min_length=1, description="One DayPlan per day.")

    @model_validator(mode="after")
    def days_count_matches_total(self) -> "LearningPath":
        if len(self.days) != self.total_days:
            # Allow partial (iterative generation), just warn rather than error
            pass
        # Ensure day numbers are sequential starting at 1
        for i, day in enumerate(self.days, start=1):
            if day.day_number != i:
                day.day_number = i  # auto-correct numbering
        return self


# ---------------------------------------------------------------------------
# Markdown conversion helpers
# ---------------------------------------------------------------------------

def resource_to_markdown(r: Resource) -> str:
    """Convert a single Resource to a markdown list item."""
    year_str = f" (Uploaded {r.year})" if r.year and r.type == "Video" else ""
    return f"- [{r.title}]({r.url}){year_str} — **Type:** {r.type} — {r.reason}"


def day_to_markdown(day: DayPlan) -> str:
    """Convert a DayPlan to a markdown section."""
    lines: List[str] = [
        f"## Day {day.day_number} — {day.topic}",
        "",
        f"**Focus:** {day.focus}",
        "",
        "### Resources",
    ]
    for r in day.resources:
        lines.append(resource_to_markdown(r))
    lines += [
        "",
        f"**Practice / Reflection:** {day.practice_task}",
        "",
        "**Learning Objectives:**",
    ]
    for obj in day.learning_objectives:
        lines.append(f"- {obj}")
    lines.append("")
    return "\n".join(lines)


def learning_path_to_markdown(lp: LearningPath, playlist_url: Optional[str] = None,
                               doc_url: Optional[str] = None, notion_url: Optional[str] = None) -> str:
    """Convert a full LearningPath to a markdown document, optionally appending resource URLs."""
    sections: List[str] = [
        f"# Learning Path: {lp.goal}",
        "",
        f"**Total Duration:** {lp.total_days} day(s)",
        "",
    ]
    for day in lp.days:
        sections.append(day_to_markdown(day))

    # Summary section
    summary_lines = ["---", "", "## Summary & Resources", ""]
    if playlist_url:
        summary_lines.append(f"- 🎧 **YouTube Playlist:** [{playlist_url}]({playlist_url})")
    if doc_url:
        summary_lines.append(f"- 📄 **Google Doc:** [{doc_url}]({doc_url})")
    if notion_url:
        summary_lines.append(f"- 📝 **Notion Page:** [{notion_url}]({notion_url})")

    if any([playlist_url, doc_url, notion_url]):
        sections.append("\n".join(summary_lines))

    return "\n".join(sections)
