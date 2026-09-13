"""Observable Behaviour Recognition Package for EduPulse AI.

Provides modular classroom observable behaviour recognition from tracked person crops.
Supports both trained PyTorch models and rule-grounded Prototype / Baseline Heuristics.
"""

from src.behaviour.behaviour_classifier import (
    BehaviourClassifier,
    BehaviourPrediction,
    BehaviourSummary,
    run_behaviour_recognition_on_tracks,
)
from src.behaviour.behaviour_labels import (
    ALL_BEHAVIOUR_CLASSES,
    BEHAVIOUR_DESCRIPTIONS,
    CLASS_HEAD_DOWN,
    CLASS_INTERACTING_WITH_PEERS,
    CLASS_LOOKING_AWAY,
    CLASS_LOOKING_TOWARD_INSTRUCTION,
    CLASS_MOBILE_DEVICE_ACTIVITY,
    CLASS_READING_WRITING,
    CLASS_UNKNOWN,
    TARGET_BEHAVIOUR_CLASSES,
    get_behaviour_bgr,
    get_behaviour_description,
    get_behaviour_hex,
    get_behaviour_rgb,
)
from src.behaviour.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_person_crop,
)

__all__ = [
    "BehaviourClassifier",
    "BehaviourPrediction",
    "BehaviourSummary",
    "run_behaviour_recognition_on_tracks",
    "preprocess_person_crop",
    "TARGET_BEHAVIOUR_CLASSES",
    "ALL_BEHAVIOUR_CLASSES",
    "BEHAVIOUR_DESCRIPTIONS",
    "CLASS_LOOKING_TOWARD_INSTRUCTION",
    "CLASS_READING_WRITING",
    "CLASS_INTERACTING_WITH_PEERS",
    "CLASS_LOOKING_AWAY",
    "CLASS_MOBILE_DEVICE_ACTIVITY",
    "CLASS_HEAD_DOWN",
    "CLASS_UNKNOWN",
    "get_behaviour_rgb",
    "get_behaviour_bgr",
    "get_behaviour_hex",
    "get_behaviour_description",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
