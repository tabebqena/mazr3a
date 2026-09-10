"""describe.py - the deterministic L0 description + importance for one event.

No model, no network, no I/O: pure functions over the metadata Frigate already
gave us. This is the PRIMARY description - it is always present and never
hallucinates - and the optional VLM caption only ever ADDS to it
(plans/event-scene-reader.md §8.1).

Importance keeps the portal's existing high/normal/low ranking behaviour: a
caption that mentions something worth looking at outranks a dull one, and a
repeat of the camera's previous caption is demoted so a bird sitting in frame
yields one interesting row rather than one per capture.
"""
import datetime

# Labels that mean "worth looking at" by default (config-overridable).
IMPORTANT_LABELS = (
    "person", "car", "truck", "motorcycle", "bicycle", "bus", "train",
    "dog", "cat", "bird", "horse", "sheep", "cow", "fire", "smoke",
)
# Labels that explicitly mean "nothing to see" by default.
LOW_LABELS = ("false_positive",)


def norm_text(text):
    """Lowercase + collapse to alphanumerics - for repeat/novelty detection."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in (text or "").lower())
    return " ".join(cleaned.split())


def _clock(epoch, tz_offset_h=0.0):
    dt = datetime.datetime.fromtimestamp(float(epoch or 0), datetime.timezone.utc)
    if tz_offset_h:
        dt = dt + datetime.timedelta(hours=float(tz_offset_h))
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _val(meta, key, default=None):
    """Read a field from a mapping OR a sqlite3.Row (Row has no .get())."""
    try:
        return meta[key]
    except (KeyError, IndexError, TypeError):
        return default


def describe_event(meta, place="", tz_offset_h=0.0):
    """One deterministic sentence for a capture, e.g.

        cam04: person (0.91) at Field 1 - 2026-09-10 09:02:01

    `meta` may be a dict OR a sqlite3.Row, needs camera/label/start_time and may
    carry score/sub_label/zones. Unknown fields degrade to an omitted clause; it
    never raises.
    """
    camera = (_val(meta, "camera") or "?").strip()
    label = (_val(meta, "label") or "something").strip() or "something"
    sub = _val(meta, "sub_label")
    score = _val(meta, "score")
    if score is None:
        score = _val(meta, "top_score")
    parts = [camera + ":", label]
    if sub:
        parts.append("({})".format(sub))
    if isinstance(score, (int, float)):
        parts.append("({:.2f})".format(float(score)))
    if place:
        parts.append("at " + str(place))
    if _val(meta, "false_positive"):
        parts.append("[false positive]")
    stamp = _clock(_val(meta, "start_time"), tz_offset_h)
    return "{} - {}".format(" ".join(parts), stamp)


def score_importance(label, confidence=None, repeat=False, place_mapped=False,
                     important_labels=IMPORTANT_LABELS, low_labels=LOW_LABELS,
                     important_bonus=60.0, low_penalty=30.0, confidence_bonus=10.0,
                     mapped_place_bonus=5.0, novelty_penalty=20.0,
                     tier_high_min=60, tier_normal_min=30):
    """(score 0-100, tier high|normal|low) for one event - no extra inference.

    Calibration: `important_bonus` alone reaches `tier_high_min`, so "the event
    mentions something worth looking at" is by itself enough to be `high`; the
    other terms only move it around that band.
    """
    text = (label or "").strip().lower()
    score = 0.0
    if text in tuple(important_labels):
        score += float(important_bonus)
    elif text in tuple(low_labels):
        score -= float(low_penalty)
    if isinstance(confidence, (int, float)):
        try:
            score += float(confidence_bonus) * max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            pass
    if place_mapped:
        score += float(mapped_place_bonus)
    if repeat:
        score -= float(novelty_penalty)
    score = max(0.0, min(100.0, score))
    value = int(round(score))
    tier = ("high" if value >= int(tier_high_min)
            else "normal" if value >= int(tier_normal_min) else "low")
    return value, tier
