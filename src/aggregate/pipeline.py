"""
pipeline.py - runs every detection layer on an image and combines them.

  Layer 1  provenance  (src/detector/provenance.py)  metadata: C2PA, EXIF, PNG
  Layer 2  classifier  (src/classifier/model.py)     trained ResNet on pixels

How they're combined:
  - Evidence of AI in the metadata is strong (C2PA "AI-generated", a prompt
    stored in the PNG, an AI tool named as the software). Nobody fakes being
    AI, so it overrides the model and the image is called AI.
  - Camera EXIF pulls the score towards real, more strongly the more complete
    it is. Full camera data (make, model and exposure settings) means the
    pixel model alone can never call the image AI - at most "uncertain".
    It isn't absolute because EXIF can be faked.
  - Otherwise the model's score decides.

Why camera EXIF counts this much: the pixel model learned real photos from
ImageNet (older, smaller photos), so modern high-resolution phone photos look
unfamiliar to it and it often over-scores them as AI.
"""

from src.detector.provenance import analyze as analyze_provenance

# Score thresholds for the verdict (score = how likely the image is AI)
AI_ABOVE = 0.70
REAL_BELOW = 0.30

# How much camera metadata counts, blended with the pixel score:
#   score = (1 - weight) * pixel_score + weight * CAMERA_PRIOR
CAMERA_PRIOR = 0.05              # what camera metadata alone "says" (very likely real)
CAMERA_WEIGHT_FULL = 0.60        # make + model + at least 2 exposure settings
CAMERA_WEIGHT_PARTIAL = 0.35     # make or model only
EXPOSURE_FIELDS = ("ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength")


def camera_evidence(exif):
    """How complete the camera metadata is: (weight, description)."""
    fields = exif.get("camera_fields", {})
    has_device = "Make" in fields or "Model" in fields
    exposure = [f for f in EXPOSURE_FIELDS if f in fields]
    device = " ".join(fields[f] for f in ("Make", "Model") if f in fields).strip()
    if has_device and "Make" in fields and "Model" in fields and len(exposure) >= 2:
        return CAMERA_WEIGHT_FULL, f"full camera data ({device}, {len(exposure)} exposure settings)"
    if has_device:
        return CAMERA_WEIGHT_PARTIAL, f"partial camera data ({device})"
    return 0.0, ""


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
    weight, camera_desc = camera_evidence(prov["exif"])
    if prov["verdict"] == "ai":
        score = max(p or 0.0, 0.97)
        reason = "metadata proves AI"
    elif p is None:
        score = CAMERA_PRIOR if weight else 0.5
        reason = "metadata only"
    elif weight:
        score = (1 - weight) * p + weight * CAMERA_PRIOR
        reason = f"pixel classifier + {camera_desc}"
        signals.append(f"Camera metadata found ({camera_desc}): counted as {weight:.0%} of "
                       f"the score, pulling it from {p:.0%} to {score:.0%} likely AI. "
                       "EXIF can be faked, so it never decides on its own.")
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
