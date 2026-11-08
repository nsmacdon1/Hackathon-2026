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
from src.detector.provenance import analyze as analyze_provenance

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
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
    const observer = new MutationObserver(() => {
        document.documentElement.classList.remove('dark');
        document.body.classList.remove('dark');
    });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

    // Stagger the card fade-in animations
    setTimeout(() => {
        const cards = document.querySelectorAll('.glass-card');
        cards.forEach((card, i) => {
            card.style.animationDelay = (i * 0.12) + 's';
        });
    }, 50);
}
"""

CUSTOM_CSS = """
/* Import Inter from Google Fonts for clean reliable rendering */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* Force light mode on every surface */
:root, html, body, .dark, body.dark, html.dark {
    --body-background-fill: #ffffff !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #fafafa !important;
    --block-background-fill: rgba(255,255,255,0.7) !important;
    --border-color-primary: rgba(0,0,0,0.06) !important;
    --body-text-color: #1a1a1a !important;
    --input-background-fill: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    color-scheme: light !important;
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes float {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%      { transform: translate(30px, -20px) scale(1.05); }
    66%      { transform: translate(-20px, 15px) scale(0.97); }
}

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(249, 115, 22, 0.15); }
    50%      { box-shadow: 0 0 35px rgba(249, 115, 22, 0.3); }
}

/* Background with soft floating blobs */
html, body {
    min-height: 100vh !important;
    background: #ffffff !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif !important;
    font-size: 17px !important;
    overflow-y: auto !important;
    scroll-behavior: smooth !important;
    margin: 0 !important;
    padding: 0 !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
}

/* Animated ambient background blobs */
body::before, body::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.5;
    z-index: -1;
    pointer-events: none;
}

body::before {
    width: 600px;
    height: 600px;
    top: -120px;
    right: -100px;
    background: radial-gradient(circle, rgba(255, 186, 116, 0.4) 0%, transparent 70%);
    animation: float 18s ease-in-out infinite;
}

body::after {
    width: 500px;
    height: 500px;
    bottom: -80px;
    left: -60px;
    background: radial-gradient(circle, rgba(254, 215, 170, 0.35) 0%, transparent 70%);
    animation: float 22s ease-in-out infinite reverse;
}

.gradio-container {
    max-width: 880px !important;
    margin: 0 auto !important;
    padding: 52px 24px 140px 24px !important;
    background: transparent !important;
}

/* Frosted glass cards */
.glass-card {
    background: rgba(255, 255, 255, 0.72) !important;
    backdrop-filter: blur(40px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
    border-radius: 28px !important;
    padding: 40px 44px !important;
    margin-bottom: 24px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.04),
                0 1px 4px rgba(0, 0, 0, 0.03),
                inset 0 1px 0 rgba(255, 255, 255, 0.8) !important;
    opacity: 0;
    animation: fadeInUp 0.65s cubic-bezier(0.22, 1, 0.36, 1) forwards !important;
    transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1),
                box-shadow 0.35s ease !important;
}

.glass-card:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.07),
                0 4px 12px rgba(0, 0, 0, 0.04) !important;
}

/* Title with solid text rendering - no broken letters */
.hero-title {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 3.4rem !important;
    font-weight: 900 !important;
    color: #ea580c !important;
    letter-spacing: -0.02em !important;
    line-height: 1.1 !important;
    margin: 0 0 14px 0 !important;
    -webkit-font-smoothing: antialiased !important;
    text-rendering: geometricPrecision !important;
}

.hero-subtitle {
    font-size: 1.15rem !important;
    color: #737373 !important;
    font-weight: 400 !important;
    line-height: 1.6 !important;
    margin: 0 !important;
    max-width: 560px !important;
}

.section-heading {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
    letter-spacing: -0.01em !important;
    margin: 0 0 20px 0 !important;
}

/* Pill badge */
.tag-pill {
    display: inline-block;
    padding: 5px 14px;
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    color: #c2410c;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 16px;
    border: 1px solid rgba(251, 146, 60, 0.25);
}

/* Orange gradient pill button with glow pulse */
button.primary, .action-pill {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9999px !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 16px 44px !important;
    cursor: pointer !important;
    min-height: 54px !important;
    box-shadow: 0 8px 24px rgba(234, 88, 12, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1) !important;
    animation: pulseGlow 3s ease-in-out infinite !important;
}

button.primary:hover, .action-pill:hover {
    background: linear-gradient(135deg, #fb923c 0%, #f97316 100%) !important;
    box-shadow: 0 12px 32px rgba(234, 88, 12, 0.4) !important;
    transform: translateY(-2px) scale(1.02) !important;
    animation: none !important;
}

button.primary:active, .action-pill:active {
    transform: translateY(0) scale(0.98) !important;
}

/* Image upload area - clean, light, inviting */
.image-container, [data-testid="image"] {
    background: #ffffff !important;
    border: 2px dashed rgba(251, 146, 60, 0.35) !important;
    border-radius: 22px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    min-height: 220px !important;
}

.image-container:hover, [data-testid="image"]:hover {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.08) !important;
}

div:has(> .upload-container), .upload-container {
    background: transparent !important;
}

/* Text inputs - light and airy */
textarea, input[type="text"] {
    background: #ffffff !important;
    border: 1.5px solid rgba(0, 0, 0, 0.08) !important;
    border-radius: 16px !important;
    color: #1a1a1a !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-size: 1.05rem !important;
    padding: 14px 18px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.12) !important;
}

label span {
    color: #525252 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* Strip default Gradio chrome */
.block, .form, .panel, .gr-box, .gr-panel, .gr-form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Accordion styling */
.accordion {
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* Status indicator dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    position: relative;
    top: -1px;
}
.status-active { background: #22c55e; box-shadow: 0 0 6px rgba(34, 197, 94, 0.5); }
.status-planned { background: #d4d4d8; }

/* Pipeline list */
.pipeline-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.pipeline-list li {
    padding: 14px 0;
    border-bottom: 1px solid rgba(0,0,0,0.04);
    font-size: 1.05rem;
    color: #404040;
    line-height: 1.5;
}
.pipeline-list li:last-child { border-bottom: none; }
.pipeline-list li strong { color: #1a1a1a; font-weight: 700; }
"""


def detect_image(image_path):
    if not image_path:
        return (
            "No Image Provided",
            "0%",
            "Please upload an image above to begin detection.",
            {},
        )

    # Convert HEIC, AVIF, WebP, TIFF to JPEG for reliable downstream analysis
    analysis_path = ensure_compatible(image_path)

    prov_result = analyze_provenance(analysis_path)
    verdict = prov_result.get("verdict", "unknown")

    if verdict == "ai":
        baked_score = "95% Baked"
        verdict_display = "Baked (AI-Generated Image)"
    elif verdict == "likely_real":
        baked_score = "15% Baked"
        verdict_display = "Real (Authentic Camera Photo)"
    else:
        baked_score = "50% Baked"
        verdict_display = "Inconclusive (Needs Pixel and Frequency Analysis)"

    signals = prov_result.get("signals", [])
    signals_display = "\n".join(f"- {s}" for s in signals) if signals else "No notable signals detected."

    raw_data = {
        "verdict": verdict,
        "c2pa": prov_result.get("c2pa", {}),
        "exif": prov_result.get("exif", {}),
        "png_text": prov_result.get("png_text", {}),
    }

    return verdict_display, baked_score, signals_display, raw_data


def build_app():
    theme = gr.themes.Default(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.zinc,
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="Real or Baked", theme=theme, css=CUSTOM_CSS, js=FORCE_LIGHT_JS) as demo:

        # Hero card
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML('<span class="tag-pill">AI Image Detector</span>')
            gr.HTML('<h1 class="hero-title">Real or Baked</h1>')
            gr.HTML(
                '<p class="hero-subtitle">A multi-layered detection pipeline that checks '
                'digital provenance, sensor physics, and generator fingerprints to figure '
                'out if an image is authentic or AI-generated.</p>'
            )

        # Upload card
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML('<h2 class="section-heading">Upload an Image</h2>')
            gr.HTML(
                '<p style="color:#737373; font-size:0.95rem; margin:0 0 12px 0;">'
                'Supports JPEG, PNG, HEIC, WebP, TIFF, AVIF, BMP, and GIF.</p>'
            )
            image_input = gr.Image(
                type="filepath",
                label="Drop an image here or click to browse",
            )
            analyze_button = gr.Button("Analyze Image", variant="primary", elem_classes=["action-pill"])

        # Verdict card
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML('<h2 class="section-heading">Detection Verdict</h2>')
            with gr.Row():
                verdict_output = gr.Textbox(
                    label="Verdict",
                    placeholder="Waiting for an image ...",
                    interactive=False,
                    scale=2,
                )
                score_output = gr.Textbox(
                    label="Baked Score",
                    placeholder="--",
                    interactive=False,
                    scale=1,
                )

        # Signals card
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML('<h2 class="section-heading">Signals and Evidence</h2>')
            signals_output = gr.Textbox(
                label="Why this verdict was given",
                placeholder="Detected fingerprints and reasoning will appear here.",
                lines=5,
                interactive=False,
            )

        # Provenance deep dive card
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML('<h2 class="section-heading">Provenance Deep Dive</h2>')
            with gr.Accordion("Inspect Raw EXIF and C2PA Data", open=False):
                details_output = gr.JSON(label="Metadata Inspection Data")

        # Pipeline architecture card
        with gr.Column(elem_classes=["glass-card"]):
            gr.HTML('<h2 class="section-heading">Detection Pipeline</h2>')
            gr.HTML("""
                <ul class="pipeline-list">
                    <li>
                        <span class="status-dot status-active"></span>
                        <strong>Layer 1: Provenance</strong>
                        : C2PA manifests, camera exposure tags, PNG prompt chunks.
                    </li>
                    <li>
                        <span class="status-dot status-planned"></span>
                        <strong>Layer 2: ML Classifier</strong>
                        : ResNet trained on synthetic vs real images.
                    </li>
                    <li>
                        <span class="status-dot status-planned"></span>
                        <strong>Layer 3: Frequency Analysis</strong>
                        : 2D Fourier transform (FFT) grid artifact detector.
                    </li>
                    <li>
                        <span class="status-dot status-planned"></span>
                        <strong>Aggregator and Explainer</strong>
                        : Fuses all layer scores and produces region heatmaps.
                    </li>
                </ul>
            """)

        # Events
        analyze_button.click(
            fn=detect_image,
            inputs=[image_input],
            outputs=[verdict_output, score_output, signals_output, details_output],
        )

        image_input.change(
            fn=detect_image,
            inputs=[image_input],
            outputs=[verdict_output, score_output, signals_output, details_output],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
