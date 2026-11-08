"""
main.py - Web application for Real or Baked.
Apple-inspired frosted glass design with staggered animations,
floating ambient blobs, and clean typography.
"""

import os
import sys
import tempfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Register HEIC/HEIF support with Pillow before anything else imports PIL
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

import gradio as gr
from PIL import Image
from src.aggregate.pipeline import analyze as analyze_pipeline

# Formats accepted by the upload widget
ACCEPTED_FORMATS = [
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
    ".bmp", ".tiff", ".tif", ".avif", ".gif",
]


def ensure_compatible(image_path):
    """Convert unsupported formats (HEIC, AVIF, etc.) to a temporary
    JPEG so that downstream analysis tools can open the file reliably.
    Returns the path to use for analysis (may be the original)."""
    ext = os.path.splitext(image_path)[1].lower()
    needs_convert = ext in {".heic", ".heif", ".avif", ".webp", ".tiff", ".tif"}
    if not needs_convert:
        return image_path

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            img.save(tmp.name, format="JPEG", quality=95, exif=img.info.get("exif", b""))
            return tmp.name
    except Exception:
        return image_path

FORCE_LIGHT_JS = """
() => {
    // Black-and-white design: always light, even if the browser prefers dark
    const strip = () => {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
    };
    strip();
    new MutationObserver(strip).observe(document.documentElement,
        { attributes: true, attributeFilter: ['class'] });
}
"""

CUSTOM_CSS = """

:root, .dark {
    --ink: #0a0a0a;
    --ink-2: #5c5c5c;
    --ink-3: #9a9a9a;
    --line: #0a0a0a;
    --line-soft: #e6e6e6;
    --paper: #ffffff;
    --paper-2: #f6f6f6;
    --sans: 'Inter', -apple-system, 'Helvetica Neue', Arial, sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, monospace;

    /* Gradio theme variables */
    --body-background-fill: var(--paper) !important;
    --background-fill-primary: var(--paper) !important;
    --background-fill-secondary: var(--paper-2) !important;
    --block-background-fill: var(--paper) !important;
    --block-border-color: var(--line-soft) !important;
    --border-color-primary: var(--line-soft) !important;
    --body-text-color: var(--ink) !important;
    --body-text-color-subdued: var(--ink-2) !important;
    --input-background-fill: var(--paper) !important;
    --color-accent: var(--ink) !important;
    --color-accent-soft: var(--paper-2) !important;
    --block-label-text-color: var(--ink-2) !important;
    --block-title-text-color: var(--ink-2) !important;
    color-scheme: light !important;
}

html, body {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: var(--sans) !important;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
    padding: 40px 28px 96px !important;
    background: var(--paper) !important;
    font-family: var(--sans) !important;
}

footer { display: none !important; }

/* Gradio pads every HTML block; our sections handle their own spacing */
.html-container { padding: 0 !important; }

/* ---------- Header ---------- */
.masthead {
    border-bottom: 2px solid var(--line);
    padding-bottom: 28px;
    margin-bottom: 8px;
}
.masthead .kicker {
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-2);
    display: flex;
    justify-content: space-between;
    margin-bottom: 18px;
}
.masthead h1 {
    font-family: var(--sans);
    font-size: clamp(2.6rem, 7vw, 4.6rem);
    font-weight: 900;
    letter-spacing: -0.045em;
    line-height: 0.95;
    margin: 0 0 16px;
    color: var(--ink);
}
.masthead p {
    font-size: 1.08rem;
    line-height: 1.55;
    color: var(--ink-2);
    max-width: 620px;
    margin: 0;
}

/* ---------- Section labels ---------- */
.section-label {
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ink);
    border-top: 1px solid var(--line);
    padding-top: 12px;
    margin: 28px 0 14px;
    display: flex;
    gap: 14px;
}
.section-label b { font-weight: 700; }
.section-label span { color: var(--ink-2); }

/* ---------- Upload ---------- */
.upload-box, .upload-box > div {
    border-radius: 2px !important;
}
.upload-box {
    border: 1.5px dashed var(--ink) !important;
    background: var(--paper-2) !important;
    min-height: 340px !important;
}
.upload-box:hover { background: #efefef !important; }

/* ---------- Button ---------- */
button.analyze-btn, .analyze-btn button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: 1.5px solid var(--ink) !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    min-height: 54px !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
button.analyze-btn:hover, .analyze-btn button:hover {
    background: var(--paper) !important;
    color: var(--ink) !important;
}

/* ---------- Verdict ---------- */
.verdict {
    border: 1.5px solid var(--line);
    padding: 28px 28px 24px;
    min-height: 340px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: var(--paper);
    color: var(--ink);
}
.verdict .v-label {
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    opacity: 0.7;
}
.verdict .v-word {
    font-family: var(--sans);
    font-size: clamp(3rem, 8vw, 5.4rem);
    font-weight: 900;
    letter-spacing: -0.05em;
    line-height: 0.9;
    margin: 18px 0 10px;
}
.verdict .v-sub {
    font-size: 1.02rem;
    opacity: 0.8;
}
.verdict.is-ai { background: var(--ink); color: var(--paper) !important; }
/* Gradio's prose styles force dark text; make everything follow the box */
.verdict * { color: inherit !important; }
.verdict.is-uncertain { border-style: dashed; }
.verdict.is-empty .v-word { color: var(--ink-3); }

.meter { margin-top: 28px; }
.meter-track {
    position: relative;
    height: 12px;
    border: 1.5px solid currentColor;
}
.meter-fill {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: currentColor;
}
.meter-tick {
    position: absolute; top: -6px; bottom: -6px; left: 50%;
    border-left: 1.5px dashed currentColor;
    opacity: 0.6;
}
.meter-scale {
    display: flex;
    justify-content: space-between;
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 10px;
}
.meter-scale b { font-weight: 700; }

/* ---------- Text outputs ---------- */
textarea, input[type="text"] {
    background: var(--paper) !important;
    border: 1px solid var(--line-soft) !important;
    border-radius: 2px !important;
    color: var(--ink) !important;
    font-family: var(--mono) !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    box-shadow: none !important;
}
textarea:focus, input[type="text"]:focus { border-color: var(--ink) !important; }

.block, .form, .panel {
    box-shadow: none !important;
    border-radius: 2px !important;
}

.accordion, .details-box {
    border: 1px solid var(--line-soft) !important;
    border-radius: 2px !important;
}

/* ---------- Pipeline ---------- */
.pipeline {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1.5px solid var(--line);
}
.pipeline > div {
    padding: 18px 18px 20px;
    border-right: 1px solid var(--line);
}
.pipeline > div:last-child { border-right: none; }
.pipeline .p-num {
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    color: var(--ink-2);
    display: flex; justify-content: space-between;
}
.pipeline .p-name {
    font-weight: 800;
    font-size: 1.05rem;
    margin: 10px 0 6px;
    letter-spacing: -0.01em;
}
.pipeline .p-desc { font-size: 0.9rem; color: var(--ink-2); line-height: 1.45; }
.pipeline .planned .p-name { color: var(--ink-2); }

@media (max-width: 760px) {
    .pipeline { grid-template-columns: 1fr 1fr; }
    .pipeline > div:nth-child(2) { border-right: none; }
    .pipeline > div:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
    .verdict, .upload-box { min-height: 260px !important; }
}
"""


def verdict_html(word, sub, score=None, kind="empty"):
    """The verdict panel: one big word, a line of detail, and a Real-to-AI meter."""
    if score is None:
        meter = ""
    else:
        pct = max(0.0, min(1.0, score)) * 100
        meter = f"""
        <div class="meter">
            <div class="meter-track">
                <div class="meter-fill" style="width:{pct:.1f}%"></div>
                <div class="meter-tick"></div>
            </div>
            <div class="meter-scale"><span>Real</span><b>{pct:.0f}% baked</b><span>AI</span></div>
        </div>"""
    return f"""
    <div class="verdict is-{kind}">
        <div>
            <div class="v-label">Verdict</div>
            <div class="v-word">{word}</div>
            <div class="v-sub">{sub}</div>
        </div>
        {meter}
    </div>"""


EMPTY_VERDICT = verdict_html("—", "Upload an image to analyze it.")


def detect_image(image_path):
    if not image_path:
        return EMPTY_VERDICT, "", {}

    # Convert HEIC, AVIF, WebP, TIFF to JPEG for reliable downstream analysis
    analysis_path = ensure_compatible(image_path)

    # Layer 1 (provenance) + Layer 2 (pixel classifier), combined
    result = analyze_pipeline(analysis_path)
    verdict = result["verdict"]
    confidence = result["confidence"]
    score = result["score"]

    if verdict == "ai":
        html = verdict_html("Baked.", f"AI-generated · {confidence} confidence", score, "ai")
    elif verdict == "real":
        html = verdict_html("Real.", f"Authentic image · {confidence} confidence", score, "real")
    else:
        html = verdict_html("Unsure.", "Could be either - see the evidence below", score, "uncertain")

    signals = result.get("signals", [])
    signals_display = "\n".join(f"→ {s}" for s in signals) if signals else "No notable signals detected."

    prov_result = result["provenance"]
    raw_data = {
        "verdict": verdict,
        "score": round(score, 4),
        "decided_by": result["reason"],
        "classifier": result["classifier"],
        "provenance_verdict": prov_result.get("verdict"),
        "c2pa": prov_result.get("c2pa", {}),
        "exif": prov_result.get("exif", {}),
        "png_text": prov_result.get("png_text", {}),
    }
    return html, signals_display, raw_data


def section(num, title, note=""):
    return gr.HTML(f'<div class="section-label"><b>{num}</b>{title}'
                   f'{f"<span>{note}</span>" if note else ""}</div>')


PIPELINE_HTML = """
<div class="pipeline">
  <div class="active">
    <div class="p-num"><span>LAYER 1</span><span>●</span></div>
    <div class="p-name">Provenance</div>
    <div class="p-desc">C2PA content credentials, camera EXIF, AI prompts hidden in PNG files.</div>
  </div>
  <div class="active">
    <div class="p-num"><span>LAYER 2</span><span>●</span></div>
    <div class="p-name">Pixel classifier</div>
    <div class="p-desc">ResNet-50 trained on real vs AI images, hardened against compression.</div>
  </div>
  <div class="planned">
    <div class="p-num"><span>LAYER 3</span><span>○</span></div>
    <div class="p-name">Frequency</div>
    <div class="p-desc">Fourier analysis of generator upsampling grids. Planned.</div>
  </div>
  <div class="active">
    <div class="p-num"><span>AGGREGATOR</span><span>●</span></div>
    <div class="p-name">Verdict</div>
    <div class="p-desc">Metadata proof overrides; otherwise the pixel score decides.</div>
  </div>
</div>
"""


def build_app():
    with gr.Blocks(title="Real or Baked") as demo:

        gr.HTML("""
            <header class="masthead">
                <div class="kicker"><span>AI image forensics</span><span>Hackathon 2026</span></div>
                <h1>Real or Baked</h1>
                <p>Upload any image. We check its digital paperwork and its pixels,
                   then tell you whether a camera or a generator made it.</p>
            </header>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                section("01", "Image", "any format")
                image_input = gr.Image(
                    type="filepath",
                    label="Drop an image here or click to browse",
                    show_label=False,
                    elem_classes=["upload-box"],
                    height=340,
                )
                analyze_button = gr.Button("Analyze image →", elem_classes=["analyze-btn"])
            with gr.Column(scale=1):
                section("02", "Verdict")
                verdict_output = gr.HTML(EMPTY_VERDICT)

        section("03", "Evidence", "why this verdict was given")
        signals_output = gr.Textbox(
            show_label=False,
            placeholder="Signals from each layer will appear here.",
            lines=6,
            interactive=False,
        )
        with gr.Accordion("Raw metadata — EXIF, C2PA, classifier output", open=False,
                          elem_classes=["details-box"]):
            details_output = gr.JSON(show_label=False)

        section("04", "Pipeline", "● active   ○ planned")
        gr.HTML(PIPELINE_HTML)

        outputs = [verdict_output, signals_output, details_output]
        analyze_button.click(fn=detect_image, inputs=[image_input], outputs=outputs)
        image_input.change(fn=detect_image, inputs=[image_input], outputs=outputs)

    return demo


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Inter"), "Helvetica Neue", "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_none,
)


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False,
               theme=THEME, css=CUSTOM_CSS, js=FORCE_LIGHT_JS)
