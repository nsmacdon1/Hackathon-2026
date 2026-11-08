"""
pipeline.py - runs every detection layer on an image and combines them.

  Layer 1  provenance  (src/detector/provenance.py)  metadata: C2PA, EXIF, PNG
  Layer 2  classifier  (src/classifier/model.py)     trained ResNet on pixels

How they're combined:
  - Evidence of AI in the metadata is strong (C2PA "AI-generated", a prompt
    stored in the PNG, an AI tool named as the software). Nobody fakes being
    AI, so it overrides the model and the image is called AI.
  - Camera EXIF only nudges the score towards real: it's easy to fake.
  - Otherwise the model's score decides.
"""

from src.detector.provenance import analyze as analyze_provenance

# Score thresholds for the verdict (score = how likely the image is AI)
AI_ABOVE = 0.70
REAL_BELOW = 0.30
# Camera EXIF moves the model's score this far towards "real"
CAMERA_EXIF_NUDGE = 0.8


def analyze(image_path):
    prov = analyze_provenance(image_path)
    signals = list(prov["signals"])

    # --- Layer 2: the model ------------------------------------------------
    try:
        from src.classifier.model import predict   # imported late: loads torch
        clf = predict(image_path)
        p = clf["prob_ai"]
        signals.append(f"Pixel classifier ({clf['model']}): {p:.0%} likely AI. "
                       "Trained on Stable Diffusion v1.4 images - less reliable "
                       "on other generators.")
    except Exception as e:
        clf, p = {"error": str(e)}, None
        signals.append(f"Pixel classifier unavailable: {e}")

    # --- Combine -----------------------------------------------------------
    if prov["verdict"] == "ai":
        score = max(p or 0.0, 0.97)
        reason = "metadata proves AI"
    elif p is None:
        score = 0.15 if prov["verdict"] == "likely_real" else 0.5
        reason = "metadata only"
    elif prov["verdict"] == "likely_real":
        score = p * CAMERA_EXIF_NUDGE
        reason = "pixel classifier, nudged towards real by camera EXIF"
    else:
        score = p
        reason = "pixel classifier"

    if score >= AI_ABOVE:
        verdict = "ai"
    elif score <= REAL_BELOW:
        verdict = "real"
    else:
        verdict = "uncertain"

    # how far from the 50/50 middle the score is
    confidence = "high" if abs(score - 0.5) >= 0.4 else "medium" if abs(score - 0.5) >= 0.2 else "low"

    return {"verdict": verdict, "score": score, "confidence": confidence,
            "reason": reason, "signals": signals,
            "provenance": prov, "classifier": clf}
