"""Observable Behaviour Class Definitions & Metadata for EduPulse AI.

Defines the six target observable learning-related behaviour categories,
the fallback uncertain state, color palettes, and strict research boundary definitions.
"""

from typing import Dict, List, Tuple
from src.config import (
    ALL_OBSERVABLE_BEHAVIOURS,
    BEHAVIOUR_COLORS,
    CLASS_HEAD_DOWN,
    CLASS_INTERACTING_WITH_PEERS,
    CLASS_LOOKING_AWAY,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_MOBILE_DEVICE_ACTIVITY,
    CLASS_READING_WRITING,
    CLASS_UNKNOWN,
    TARGET_OBSERVABLE_BEHAVIOURS,
    UNKNOWN_BEHAVIOUR,
)

# The 6 defined research target classes
TARGET_BEHAVIOUR_CLASSES: List[str] = list(TARGET_OBSERVABLE_BEHAVIOURS)

# All valid classes including the fallback unknown/uncertain state
ALL_BEHAVIOUR_CLASSES: List[str] = list(ALL_OBSERVABLE_BEHAVIOURS)

# Observable visual evidence and research boundaries for each category
BEHAVIOUR_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    CLASS_LOOKING_TOWARD_INSTRUCTION: {
        "observable_evidence": "Body or head oriented toward the instructor, presentation screen, or front blackboard area.",
        "scientific_boundary": "Does not prove cognitive attention, motivation, or comprehension.",
    },
    CLASS_READING_WRITING: {
        "observable_evidence": "Head angled downward toward a notebook/book/paper with visible hand or arm posture associated with writing or reading.",
        "scientific_boundary": "Does not infer academic effort or learning quality.",
    },
    CLASS_INTERACTING_WITH_PEERS: {
        "observable_evidence": "Head or upper body oriented toward a nearby student, with proximal collaborative or conversational posture.",
        "scientific_boundary": "Does not infer conversation topic, social valence, or classroom disruption.",
    },
    CLASS_LOOKING_AWAY: {
        "observable_evidence": "Head or torso oriented substantially away from instructional activity toward windows, doors, or peripheral space.",
        "scientific_boundary": "Does not label or equate to boredom, apathy, or disengagement.",
    },
    CLASS_MOBILE_DEVICE_ACTIVITY: {
        "observable_evidence": "Visible handheld mobile phone or tablet, or gaze directed downward into a handheld object in lap/desk region.",
        "scientific_boundary": "Requires visual evidence; does not imply educational vs. non-educational content.",
    },
    CLASS_HEAD_DOWN: {
        "observable_evidence": "Head resting directly on desk, arms folded under head, or face hidden downward without reading/writing movement.",
        "scientific_boundary": "Does not conclude sleep, exhaustion, depression, or disengagement.",
    },
    CLASS_UNKNOWN: {
        "observable_evidence": "Severe occlusion, blurry crop, low resolution, ambiguous posture, or prediction below confidence threshold.",
        "scientific_boundary": "Preserves research rigor by rejecting forced false classifications.",
    },
}


def get_behaviour_rgb(label: str) -> Tuple[int, int, int]:
    """Return the RGB color tuple for a given behaviour class."""
    if label in BEHAVIOUR_COLORS:
        return BEHAVIOUR_COLORS[label]["rgb"]
    return BEHAVIOUR_COLORS[CLASS_UNKNOWN]["rgb"]


def get_behaviour_bgr(label: str) -> Tuple[int, int, int]:
    """Return the BGR color tuple (OpenCV format) for a given behaviour class."""
    r, g, b = get_behaviour_rgb(label)
    return (b, g, r)


def get_behaviour_hex(label: str) -> str:
    """Return the Hex color string for a given behaviour class."""
    if label in BEHAVIOUR_COLORS:
        return BEHAVIOUR_COLORS[label]["hex"]
    return BEHAVIOUR_COLORS[CLASS_UNKNOWN]["hex"]


def get_behaviour_description(label: str) -> Dict[str, str]:
    """Return the evidence and scientific boundary for a given behaviour class."""
    return BEHAVIOUR_DESCRIPTIONS.get(
        label,
        {
            "observable_evidence": "Unspecified observable evidence.",
            "scientific_boundary": "Strict observable scope applies.",
        },
    )
