"""
provenance.py - Layer 1 of the AI-image detector.
 
Checks the "paperwork" attached to an image file, not the pixels:
  1. C2PA Content Credentials (signed provenance manifests)
  2. EXIF metadata (camera make/model, software, exposure settings)
  3. PNG text chunks (where Stable Diffusion / ComfyUI tools store prompts)
 
Install:  pip install c2pa-python pillow
Run:      python provenance.py some_image.jpg
"""
 
import json
import sys
 
try:
    import c2pa
except ImportError:
    c2pa = None
from PIL import Image
from PIL.ExifTags import TAGS
 
# ---------------------------------------------------------------------------
# Known signals
# ---------------------------------------------------------------------------
 
# IPTC "digital source type" codes that C2PA manifests use to say how an image
# was made. "trainedAlgorithmicMedia" is the official code for generative AI.
AI_SOURCE_TYPES = {
    "trainedAlgorithmicMedia": "fully AI-generated",
    "compositeWithTrainedAlgorithmicMedia": "edited with generative AI",
    "algorithmicMedia": "algorithmically generated",
}
 
# Words that show up in the EXIF "Software" field or C2PA "claim_generator"
# when an AI tool wrote the file. Extend this list as you find more.
AI_SOFTWARE_KEYWORDS = [
    "dall-e", "dalle", "openai", "midjourney", "stable diffusion",
    "stablediffusion", "comfyui", "automatic1111", "invokeai", "firefly",
    "imagen", "gemini", "flux", "leonardo", "ideogram", "novelai",
]
 
# PNG text-chunk keys that local generators write (the prompt + settings).
AI_PNG_KEYS = {"parameters", "prompt", "workflow", "sd-metadata",
               "invokeai_metadata", "dream"}
 
 
# ---------------------------------------------------------------------------
# 1. C2PA Content Credentials
# ---------------------------------------------------------------------------
 
def check_c2pa(path):
    """Read a C2PA manifest if the file has one.
 
    Returns a dict like:
      {"present": True, "valid": True, "generator": "...",
       "ai_source": "fully AI-generated" or None}
    """
    if c2pa is None:
        return {"present": False}

    try:
        reader = c2pa.Reader(path)
    except Exception:
        # No manifest embedded (the common case for most images today).
        return {"present": False}
 
    store = json.loads(reader.json())
    active_id = store.get("active_manifest")
    manifest = store.get("manifests", {}).get(active_id, {})
 
    result = {
        "present": True,
        # validation_state is "Valid"/"Trusted" when the signature checks out
        # and the image hasn't been altered since signing.
        "valid": store.get("validation_state") in ("Valid", "Trusted"),
        "generator": manifest.get("claim_generator", ""),
        "ai_source": None,
    }
 
    # Walk every assertion looking for a digitalSourceType.
    # It usually lives inside a "c2pa.actions" assertion, on an action
    # like {"action": "c2pa.created", "digitalSourceType": ".../trainedAlgorithmicMedia"}.
    blob = json.dumps(manifest.get("assertions", []))
    for code, meaning in AI_SOURCE_TYPES.items():
        if code in blob:
            result["ai_source"] = meaning
            break   # dict order puts the most specific code first
 
    reader.close()
    return result
 
 
# ---------------------------------------------------------------------------
# 2. EXIF metadata
# ---------------------------------------------------------------------------
 
def check_exif(img):
    """Pull the EXIF fields that matter for authenticity."""
    exif = img.getexif()
    # getexif() returns numeric tag IDs; translate them to readable names.
    tags = {TAGS.get(k, k): v for k, v in exif.items()}
 
    # Exposure details (ISO, shutter, aperture) live in a sub-block (IFD 0x8769).
    try:
        sub = exif.get_ifd(0x8769)
        tags.update({TAGS.get(k, k): v for k, v in sub.items()})
    except Exception:
        pass
 
    camera_fields = ["Make", "Model", "ExposureTime", "FNumber",
                     "ISOSpeedRatings", "FocalLength", "DateTimeOriginal"]
    found = {f: str(tags[f]) for f in camera_fields if f in tags}
 
    return {
        "has_exif": bool(tags),
        "camera_fields": found,
        "software": str(tags.get("Software", "")),
    }
 
 
# ---------------------------------------------------------------------------
# 3. PNG text chunks
# ---------------------------------------------------------------------------
 
def check_png_text(img):
    """Stable Diffusion UIs embed the prompt in PNG tEXt/iTXt chunks.
    Pillow exposes those in img.info."""
    if img.format != "PNG":
        return {}
    return {k: str(v)[:200] for k, v in img.info.items()
            if k.lower() in AI_PNG_KEYS}
 
 
# ---------------------------------------------------------------------------
# Combine everything into a verdict
# ---------------------------------------------------------------------------
 
def analyze(path):
    signals = []   # human-readable reasons, shown in the UI
    verdict = "unknown"
 
    # --- C2PA: the strongest evidence, because it is cryptographically signed
    c2 = check_c2pa(path)
    if c2["present"]:
        if c2["ai_source"]:
            verdict = "ai"
            signals.append(f"C2PA manifest says: {c2['ai_source']}")
        elif any(k in c2["generator"].lower() for k in AI_SOFTWARE_KEYWORDS):
            verdict = "ai"
            signals.append(f"C2PA manifest written by AI tool: {c2['generator']}")
        else:
            signals.append(f"C2PA manifest present (generator: {c2['generator']})")
        if not c2["valid"]:
            signals.append("C2PA signature did not validate (file may be altered)")
 
    with Image.open(path) as img:
        exif = check_exif(img)
        png = check_png_text(img)
 
    # --- PNG generator chunks: very strong when present
    if png:
        verdict = "ai"
        signals.append(f"PNG contains generator metadata: {', '.join(png)}")
 
    # --- EXIF software field
    sw = exif["software"].lower()
    if any(k in sw for k in AI_SOFTWARE_KEYWORDS):
        verdict = "ai"
        signals.append(f"EXIF Software field names an AI tool: {exif['software']}")
 
    # --- Camera EXIF: suggests a real camera, but is easy to fake
    cam = exif["camera_fields"]
    if verdict == "unknown" and "Make" in cam and len(cam) >= 3:
        verdict = "likely_real"
        signals.append("Camera EXIF present: "
                       + ", ".join(f"{k}={v}" for k, v in cam.items()))
    elif not exif["has_exif"]:
        signals.append("No EXIF metadata (common for AI images, but also for "
                       "screenshots and images re-saved by social media)")
 
    if verdict == "unknown":
        signals.append("Metadata is inconclusive - pass to the ML classifier")
 
    return {"verdict": verdict, "signals": signals,
            "c2pa": c2, "exif": exif, "png_text": png}
 
 
if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = analyze(p)
        print(f"\n{p}\n  verdict: {r['verdict']}")
        for s in r["signals"]:
            print(f"   - {s}")
 