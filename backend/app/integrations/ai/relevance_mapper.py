"""
relevance_mapper.py - Configurable Category-Aware Object Relevance Mapping
===========================================================================
Maps charitable project categories and milestone themes to expected COCO/YOLO object classes.
Provides explainable relevance scores and transparent limitation reporting
when a category's activities exceed the model's vocabulary.
"""

from typing import Dict, List, Set, Optional, Any


PROJECT_CATEGORY_RELEVANCE_MAP: Dict[str, Set[str]] = {
    "WATER_SANITATION": {"bottle", "sink", "toilet", "cup", "bowl", "boat", "fire hydrant", "person"},
    "EDUCATION": {"book", "chair", "dining table", "backpack", "laptop", "keyboard", "mouse", "person", "cell phone", "clock"},
    "ENVIRONMENT": {"potted plant", "bird", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "person"},
    "HEALTHCARE": {"bed", "person", "bottle", "chair", "sink", "refrigerator"},
    "SHELTER_HOUSING": {"chair", "bed", "couch", "truck", "car", "person", "suitcase", "backpack", "bench"},
    "RENEWABLE_ENERGY": {"truck", "car", "person", "traffic light", "potted plant", "chair"},
    "COMMUNITY_DEVELOPMENT": {"person", "bicycle", "motorcycle", "car", "truck", "bus", "bench", "chair", "potted plant", "book"}
}

# Aliases for fuzzy matching from free-form titles or descriptions
CATEGORY_ALIASES: Dict[str, str] = {
    "water": "WATER_SANITATION",
    "well": "WATER_SANITATION",
    "pump": "WATER_SANITATION",
    "sanitation": "WATER_SANITATION",
    "solar": "RENEWABLE_ENERGY",
    "energy": "RENEWABLE_ENERGY",
    "school": "EDUCATION",
    "education": "EDUCATION",
    "book": "EDUCATION",
    "student": "EDUCATION",
    "tree": "ENVIRONMENT",
    "forest": "ENVIRONMENT",
    "reforestation": "ENVIRONMENT",
    "plant": "ENVIRONMENT",
    "clinic": "HEALTHCARE",
    "medical": "HEALTHCARE",
    "health": "HEALTHCARE",
    "shelter": "SHELTER_HOUSING",
    "housing": "SHELTER_HOUSING",
    "tent": "SHELTER_HOUSING",
}


def resolve_expected_classes(
    category_name: Optional[str] = None,
    milestone_title: Optional[str] = None,
    project_description: Optional[str] = None
) -> Set[str]:
    """
    Resolves the set of expected object labels based on project context.
    Returns general classes if no specific category matches.
    """
    matched_classes: Set[str] = set()

    # Direct match on category
    if category_name:
        import re
        clean_cat = re.sub(r'[^a-zA-Z0-9]+', '_', category_name.strip()).strip('_').upper()
        for key in PROJECT_CATEGORY_RELEVANCE_MAP:
            if key == clean_cat or key in clean_cat or clean_cat in key:
                matched_classes.update(PROJECT_CATEGORY_RELEVANCE_MAP[key])
        if not matched_classes:
            cat_lower = category_name.lower()
            for key, val in PROJECT_CATEGORY_RELEVANCE_MAP.items():
                tokens = [t.lower() for t in key.split("_") if t]
                if any(tok in cat_lower for tok in tokens):
                    matched_classes.update(val)

    # Text match on milestone title or description
    context_text = f"{milestone_title or ''} {project_description or ''}".lower()
    for keyword, cat_key in CATEGORY_ALIASES.items():
        if keyword in context_text:
            matched_classes.update(PROJECT_CATEGORY_RELEVANCE_MAP[cat_key])

    # Fallback default classes
    if not matched_classes:
        matched_classes = {"person", "car", "truck", "bench", "chair", "potted plant"}

    return matched_classes


def filter_relevant_detections(
    detections: List[Dict[str, Any]],
    expected_classes: Set[str]
) -> List[Dict[str, Any]]:
    """
    Annotates each detection with relevance to the project category.
    """
    result = []
    for det in detections:
        det_copy = dict(det)
        label = str(det_copy.get("label", "")).lower()
        det_copy["is_project_relevant"] = label in expected_classes
        result.append(det_copy)
    return result
