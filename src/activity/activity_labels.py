"""Teaching Activity Labels and Metadata Module for EduPulse AI.

Provides canonical target teaching activity class constants, instructional
descriptions, and color representations for Feature 10 (Teaching Activity Analysis).

CRITICAL PEDAGOGICAL BOUNDARY:
Teaching activity classification describes the instructional classroom setting
(e.g., Lecture, Discussion, Problem-solving, Presentation). It does NOT measure
or describe internal student mental states, motivation, concentration, comprehension,
boredom, intelligence, or emotional states.
"""

from typing import Dict, List, Tuple

from src.config import (
    ACTIVITY_COLORS,
    ACTIVITY_DISCUSSION,
    ACTIVITY_LECTURE,
    ACTIVITY_PRESENTATION,
    ACTIVITY_PROBLEM_SOLVING,
    TARGET_TEACHING_ACTIVITIES,
)

# Pedagogical definitions of the instructional modalities
ACTIVITY_DESCRIPTIONS: Dict[str, str] = {
    ACTIVITY_LECTURE: (
        "Instructor-led instructional delivery where information, concepts, and explanations "
        "are presented to the whole classroom, typically characterized by teacher speech, slides, or board demonstration."
    ),
    ACTIVITY_DISCUSSION: (
        "Interactive verbal exchange involving student-to-student or teacher-to-student dialogues, "
        "debates, whole-group question-and-answer, or structured collaborative conversations."
    ),
    ACTIVITY_PROBLEM_SOLVING: (
        "Task-centered active learning where students work individually or collaboratively on exercises, "
        "computational problems, case studies, worksheets, or laboratory tasks."
    ),
    ACTIVITY_PRESENTATION: (
        "Formal or informal delivery where designated students or guest speakers present projects, "
        "solutions, findings, or demonstrations to peers or the instructor."
    ),
}

ACTIVITY_SOURCE_MANUAL = "manual"
ACTIVITY_SOURCE_METADATA = "metadata"
VALID_ANNOTATION_SOURCES = [ACTIVITY_SOURCE_MANUAL, ACTIVITY_SOURCE_METADATA]

ACTIVITY_UNKNOWN = "Unannotated"


def get_activity_hex(activity_name: str) -> str:
    """Retrieve canonical hex color string for a teaching activity.

    Args:
        activity_name: Name of the teaching activity.

    Returns:
        Hex color string (e.g., '#2980B9').
    """
    if activity_name in ACTIVITY_COLORS:
        return ACTIVITY_COLORS[activity_name]["hex"]
    return "#7F8C8D"  # Neutral gray fallback


def get_activity_rgb(activity_name: str) -> Tuple[int, int, int]:
    """Retrieve canonical RGB color tuple for a teaching activity.

    Args:
        activity_name: Name of the teaching activity.

    Returns:
        RGB tuple of integers (0-255).
    """
    if activity_name in ACTIVITY_COLORS:
        return ACTIVITY_COLORS[activity_name]["rgb"]
    return (127, 140, 141)  # Neutral gray fallback


def get_activity_description(activity_name: str) -> str:
    """Retrieve pedagogical description for a teaching activity.

    Args:
        activity_name: Name of the teaching activity.

    Returns:
        Description string.
    """
    return ACTIVITY_DESCRIPTIONS.get(
        activity_name,
        "Unannotated or general classroom interval without a designated instructional activity modality.",
    )
