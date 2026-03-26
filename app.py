"""
ClaimSense Gradio application entry point.
Provides a web UI for uploading or pasting claim documents and
displays the structured ClaimBrief as both raw JSON and a formatted card.

Run:
    python app.py            # interactive mode
    python app.py --demo     # auto-load the first sample claim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

import os as _os
_os.environ.setdefault("GRADIO_DEFAULT_LANGUAGE", "en")

# Ensure the project root is on sys.path when running app.py directly
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.classify import classify_claim
from pipeline.compliance import check_compliance
from pipeline.extract import extract_entities
from pipeline.ingest import full_text, ingest
from pipeline.rag import retrieve_similar
from pipeline.summarize import build_claim_brief
from models.schemas import ClaimBrief

_SAMPLE_DIR = _ROOT / "data" / "sample_claims"

# ── Risk level → badge colour ─────────────────────────────────────────────────
_RISK_COLOURS: dict[str, str] = {
    "Routine": "#22c55e",      # green
    "Needs Review": "#f59e0b", # amber
    "Escalate": "#ef4444",     # red
}


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(source: str | Path) -> ClaimBrief:
    """Execute the full ClaimSense pipeline on a document source."""
    chunks = ingest(source)
    text = full_text(chunks)
    entities = extract_entities(text)
    classification = classify_claim(text, entities)
    compliance = check_compliance(entities, classification)
    similar = retrieve_similar(text)
    return build_claim_brief(chunks, entities, classification, compliance, similar)


# ── Gradio handlers ───────────────────────────────────────────────────────────

def process_file(file_obj: gr.File | None) -> tuple[str, str]:
    """Handle uploaded file (PDF or .txt)."""
    if file_obj is None:
        return _err("No file uploaded.")
    try:
        brief = run_pipeline(Path(file_obj.name))
        return _render(brief)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def process_text(text: str) -> tuple[str, str]:
    """Handle pasted plain text."""
    if not text.strip():
        return _err("Please paste claim text or upload a file.")
    try:
        brief = run_pipeline(text)
        return _render(brief)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))


def _err(msg: str) -> tuple[str, str]:
    return f'{{"error": "{msg}"}}', f"<p style='color:red'>{msg}</p>"


# ── Card renderer ─────────────────────────────────────────────────────────────

def _render(brief: ClaimBrief) -> tuple[str, str]:
    """Return (raw JSON string, HTML card string)."""
    json_str = json.dumps(
        brief.model_dump(exclude={"extracted_entities"}), indent=2, ensure_ascii=False
    )
    colour = _RISK_COLOURS.get(brief.risk_level, "#6b7280")
    missing_html = (
        "<ul>" + "".join(f"<li>{f}</li>" for f in brief.missing_fields) + "</ul>"
        if brief.missing_fields
        else "<p style='color:#22c55e'>All required fields present ✓</p>"
    )
    rag_html = "".join(f"<li>{c}</li>" for c in brief.rag_similar_cases)
    parties_html = ", ".join(brief.parties_involved) or "—"

    card = f"""
<div style="font-family:sans-serif;max-width:700px;padding:16px;border:1px solid #e5e7eb;border-radius:12px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <h2 style="margin:0">{brief.claim_type} — {brief.claim_subtype}</h2>
    <span style="background:{colour};color:#fff;padding:4px 12px;border-radius:999px;font-weight:bold">
      {brief.risk_level} ({brief.risk_score}/100)
    </span>
  </div>
  <hr>
  <table style="width:100%;border-collapse:collapse">
    <tr><td style="width:40%;padding:4px;font-weight:bold">Incident Date</td>
        <td style="padding:4px">{brief.incident_date}</td></tr>
    <tr><td style="padding:4px;font-weight:bold">Claimed Amount</td>
        <td style="padding:4px">{brief.claimed_amount}</td></tr>
    <tr><td style="padding:4px;font-weight:bold">Policy Number</td>
        <td style="padding:4px">{brief.policy_number or '—'}</td></tr>
    <tr><td style="padding:4px;font-weight:bold">Parties Involved</td>
        <td style="padding:4px">{parties_html}</td></tr>
  </table>
  <hr>
  <h3>Adjuster Brief</h3>
  <blockquote style="border-left:4px solid {colour};margin:0;padding:8px 16px;background:#f9fafb">
    {brief.adjuster_brief}
  </blockquote>
  <h3>Risk Reasoning</h3>
  <p>{brief.risk_reasoning}</p>
  <h3>Compliance Checklist</h3>
  {missing_html}
  <h3>Similar Historical Cases</h3>
  <ul>{rag_html}</ul>
</div>
"""
    return json_str, card


# ── Gradio UI definition ──────────────────────────────────────────────────────

_FORCE_EN_JS = """
() => {
    Object.defineProperty(navigator, 'language', {get: () => 'en-US', configurable: true});
    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en'], configurable: true});
}
"""

def _build_ui(demo_text: str = "") -> gr.Blocks:
    with gr.Blocks(title="ClaimSense – AI Claim Triage", js=_FORCE_EN_JS) as app:
        gr.Markdown("# ClaimSense — AI-Powered Insurance Claim Triage")
        gr.Markdown(
            "Upload a claim document (PDF or .txt) **or** paste the claim text below. "
            "The system will extract entities, score risk, check compliance, and retrieve "
            "similar historical cases."
        )

        with gr.Tab("Upload File"):
            file_input = gr.File(
                label="Upload PDF or .txt", file_types=[".pdf", ".txt"]
            )
            file_btn = gr.Button("Analyse Claim", variant="primary")

        with gr.Tab("Paste Text"):
            text_input = gr.Textbox(
                label="Claim Text",
                lines=14,
                placeholder="Paste the full claim document text here…",
                value=demo_text,
            )
            text_btn = gr.Button("Analyse Claim", variant="primary")

        with gr.Row():
            json_out = gr.Code(label="Structured JSON Output", language="json")
            card_out = gr.HTML(label="Adjuster Card View")

        file_btn.click(fn=process_file, inputs=[file_input], outputs=[json_out, card_out])
        text_btn.click(fn=process_text, inputs=[text_input], outputs=[json_out, card_out])

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def _load_demo_text() -> str:
    samples = sorted(_SAMPLE_DIR.glob("*.txt"))
    if samples:
        return samples[0].read_text(encoding="utf-8")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="ClaimSense Gradio app")
    parser.add_argument(
        "--demo", action="store_true", help="Auto-load the first sample claim"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # On Hugging Face Spaces the SPACE_ID env var is set; bind to 0.0.0.0 so
    # the internal proxy can reach the app.
    on_spaces = bool(_os.environ.get("SPACE_ID"))

    demo_text = _load_demo_text() if args.demo else ""
    ui = _build_ui(demo_text=demo_text)
    ui.launch(
        server_name="0.0.0.0" if on_spaces else "127.0.0.1",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
