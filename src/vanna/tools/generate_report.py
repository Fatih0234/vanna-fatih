"""Tool for generating an HTML slide report from chat-derived structure."""

from __future__ import annotations

import html
import json
import uuid
from datetime import datetime
from typing import List, Optional, Type

from pydantic import BaseModel, Field

from vanna.components import ArtifactComponent, SimpleTextComponent, UiComponent
from vanna.core.tool import Tool, ToolContext, ToolResult

from .file_system import FileSystem, LocalFileSystem


def _render_inline(text: str) -> str:
    escaped = html.escape(text)
    # Basic inline formatting (best-effort, safe after escaping)
    escaped = escaped.replace("`", "&#96;")
    # Bold: **text**
    escaped = escaped.replace("**", "\x00")  # temporary marker
    parts = escaped.split("\x00")
    for i in range(1, len(parts), 2):
        parts[i] = f"<strong>{parts[i]}</strong>"
    escaped = "".join(parts)
    return escaped


def render_basic_markdown(markdown_text: str) -> str:
    """Render a small safe subset of Markdown into HTML.

    Supported:
    - Headings (#, ##, ###)
    - Paragraphs
    - Unordered lists (- item)
    - Inline bold (**text**)

    Everything else is treated as plain text.
    """
    lines = (markdown_text or "").strip().splitlines()
    if not lines:
        return ""

    blocks: List[str] = []
    list_items: List[str] = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            items_html = "".join(f"<li>{_render_inline(i)}</li>" for i in list_items)
            blocks.append(f"<ul>{items_html}</ul>")
            list_items = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            flush_list()
            continue

        if line.startswith("### "):
            flush_list()
            blocks.append(f"<h4>{_render_inline(line[4:])}</h4>")
            continue
        if line.startswith("## "):
            flush_list()
            blocks.append(f"<h3>{_render_inline(line[3:])}</h3>")
            continue
        if line.startswith("# "):
            flush_list()
            blocks.append(f"<h2>{_render_inline(line[2:])}</h2>")
            continue

        if line.lstrip().startswith("- "):
            list_items.append(line.lstrip()[2:])
            continue

        flush_list()
        blocks.append(f"<p>{_render_inline(line)}</p>")

    flush_list()
    return "\n".join(blocks)


class ChartEmbedRef(BaseModel):
    """Reference to a saved chart artifact to embed in the report."""

    chart_html_file: str = Field(description="HTML snippet file produced by visualize_data")
    caption: Optional[str] = Field(
        default=None, description="Optional caption explaining the chart"
    )


class ReportSlide(BaseModel):
    """A single slide (1 idea per page)."""

    title: str = Field(description="Slide title (one clear idea)")
    narrative: str = Field(
        description="Narrative explanation in Markdown (why this matters, what it shows)"
    )
    charts: List[ChartEmbedRef] = Field(
        default_factory=list,
        description="Charts to embed on this slide (if any)",
    )


class GenerateReportArgs(BaseModel):
    """Arguments for generating an HTML report deck."""

    report_title: str = Field(description="Report title")
    subtitle: Optional[str] = Field(
        default=None, description="Optional subtitle (audience/purpose)"
    )
    slides: List[ReportSlide] = Field(
        description="Slides to include (1 idea per slide)"
    )
    include_methods_slide: bool = Field(
        default=True,
        description="Whether to include a brief Data/Methods slide at the end",
    )


class GenerateReportTool(Tool[GenerateReportArgs]):
    """Tool that produces an HTML slide deck and saves it in the session file store."""

    def __init__(self, file_system: Optional[FileSystem] = None):
        self.file_system = file_system or LocalFileSystem()

    @property
    def name(self) -> str:
        return "generate_report"

    @property
    def description(self) -> str:
        return (
            "Generate a slide-style HTML report (1 idea per slide) and save it as an HTML file. "
            "Use charts previously created by visualize_data by referencing their chart_html_file."
        )

    def get_args_schema(self) -> Type[GenerateReportArgs]:
        return GenerateReportArgs

    async def execute(self, context: ToolContext, args: GenerateReportArgs) -> ToolResult:
        report_id = f"report_{uuid.uuid4().hex[:8]}"
        report_filename = f"reports/{report_id}.html"

        embedded_chart_files: List[str] = []

        slide_sections: List[str] = []
        for index, slide in enumerate(args.slides, start=1):
            charts_html: List[str] = []
            for chart_ref in slide.charts:
                embedded_chart_files.append(chart_ref.chart_html_file)
                chart_html = await self.file_system.read_file(
                    chart_ref.chart_html_file, context
                )

                caption_html = (
                    f"<div class='chart-caption'>{render_basic_markdown(chart_ref.caption)}</div>"
                    if chart_ref.caption
                    else ""
                )

                charts_html.append(
                    "<div class='chart-block'>"
                    + chart_html
                    + caption_html
                    + "</div>"
                )

            slide_sections.append(
                f"""
                <section class="slide" data-slide="{index}">
                  <div class="slide-inner">
                    <h2 class="slide-title">{html.escape(slide.title)}</h2>
                    <div class="slide-narrative">{render_basic_markdown(slide.narrative)}</div>
                    {''.join(charts_html)}
                  </div>
                </section>
                """.strip()
            )

        methods_slide = ""
        if args.include_methods_slide:
            methods_slide = f"""
            <section class="slide" data-slide="methods">
              <div class="slide-inner">
                <h2 class="slide-title">Data &amp; Methods</h2>
                <div class="slide-narrative">
                  <p>This report is generated from your chat session and the underlying SQL query results and charts.</p>
                  <ul>
                    <li>Charts are captured at the time they were created in the chat.</li>
                    <li>Interpretations are narrative summaries intended for sharing with colleagues.</li>
                  </ul>
                </div>
              </div>
            </section>
            """.strip()

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        subtitle_html = (
            f"<div class='subtitle'>{html.escape(args.subtitle)}</div>"
            if args.subtitle
            else ""
        )

        deck_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(args.report_title)}</title>
  <style>
    :root {{
      --bg: #0b1220;
      --card: #111a2e;
      --text: #eef2ff;
      --muted: #b7c1d6;
      --accent: #15a8a8;
      --border: rgba(255,255,255,0.10);
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      background: var(--bg);
      color: var(--text);
    }}
    .deck {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 28px 18px 56px;
    }}
    .slide {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 26px 26px 24px;
      margin: 0 0 18px 0;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      page-break-after: always;
    }}
    .slide-inner {{
      display: grid;
      gap: 14px;
    }}
    .cover {{
      padding: 34px 30px 30px;
    }}
    .report-title {{
      font-size: 34px;
      line-height: 1.1;
      margin: 0 0 10px 0;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 16px;
      margin-top: 6px;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 16px;
    }}
    .slide-title {{
      font-size: 22px;
      margin: 0;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}
    .slide-narrative p, .slide-narrative li {{
      color: var(--text);
      opacity: 0.92;
      line-height: 1.55;
      margin: 0 0 10px 0;
    }}
    .slide-narrative ul {{
      margin: 8px 0 0 18px;
      padding: 0;
    }}
    .slide-narrative h3, .slide-narrative h4 {{
      margin: 10px 0 8px;
    }}
    .chart-block {{
      margin-top: 6px;
      padding: 14px;
      border-radius: 12px;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      overflow: hidden;
    }}
    .chart-caption {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    @media print {{
      body {{ background: #ffffff; color: #111; }}
      .slide {{
        background: #ffffff;
        color: #111;
        border: 1px solid #e5e7eb;
        box-shadow: none;
        border-radius: 10px;
      }}
      .slide-narrative p, .slide-narrative li {{ color: #111; }}
      .chart-block {{ background: #ffffff; border: 1px solid #e5e7eb; }}
      .subtitle, .meta, .chart-caption {{ color: #374151; }}
    }}
  </style>
</head>
<body>
  <main class="deck">
    <section class="slide cover" data-slide="cover">
      <div class="slide-inner">
        <h1 class="report-title">{html.escape(args.report_title)}</h1>
        {subtitle_html}
        <div class="meta">Generated {generated_at}</div>
      </div>
    </section>
    {''.join(slide_sections)}
    {methods_slide}
  </main>
</body>
</html>
"""

        await self.file_system.write_file(
            report_filename, deck_html, context, overwrite=True
        )

        report_marker = "VANNA_REPORT_ARTIFACT " + json.dumps(
            {
                "id": report_id,
                "report_title": args.report_title,
                "report_file": report_filename,
                "embedded_chart_html_files": embedded_chart_files,
            },
            ensure_ascii=False,
        )

        user_msg = (
            f"Report generated and saved as `{report_filename}`. "
            "You can print it to PDF from the browser (Print → Save as PDF)."
        )

        return ToolResult(
            success=True,
            result_for_llm=f"{user_msg}\n\n{report_marker}",
            ui_component=UiComponent(
                rich_component=ArtifactComponent(
                    artifact_type="html",
                    content=deck_html,
                    title=args.report_title,
                    description="Slide-style HTML report (printable)",
                ),
                simple_component=SimpleTextComponent(text=user_msg),
            ),
            metadata={
                "report_id": report_id,
                "report_file": report_filename,
                "embedded_chart_html_files": embedded_chart_files,
            },
        )

