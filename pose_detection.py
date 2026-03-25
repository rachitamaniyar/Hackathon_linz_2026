import math
from typing import Any


NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def detect_pose_from_landmarks(landmarks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not landmarks or len(landmarks) < 29:
        return None

    candidates = [
        score_prayer_pose(landmarks),
        score_mountain_pose(landmarks),
        score_raised_arms_pose(landmarks),
        score_forward_fold_pose(landmarks),
        score_low_lunge_pose(landmarks),
        score_plank_pose(landmarks),
        score_cobra_pose(landmarks),
        score_downward_dog_pose(landmarks),
    ]

    best = max(candidates, key=lambda item: item["confidence"])
    return best if best["confidence"] >= 0.35 else None


def score_prayer_pose(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    wrist_center = midpoint(landmarks, LEFT_WRIST, RIGHT_WRIST)
    wrists_distance = distance(landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST])

    penalties = [
        linear_penalty(abs(shoulder_center["x"] - hip_center["x"]), 0.06, 0.34),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 175, 12, 0.18),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 175, 12, 0.18),
        linear_penalty(wrists_distance, 0.10, 0.22),
        linear_penalty(abs(wrist_center["y"] - midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)["y"]), 0.18, 0.10),
    ]
    return pack("Prayer", 1 - sum(penalties))


def score_mountain_pose(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    wrists_above_shoulders = (
        landmarks[LEFT_WRIST]["y"] < landmarks[LEFT_SHOULDER]["y"]
        and landmarks[RIGHT_WRIST]["y"] < landmarks[RIGHT_SHOULDER]["y"]
    )

    penalties = [
        linear_penalty(abs(shoulder_center["x"] - hip_center["x"]), 0.06, 0.36),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 175, 12, 0.20),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 175, 12, 0.20),
        0.18 if wrists_above_shoulders else 0.0,
    ]
    return pack("Standing-Mountain", 1 - sum(penalties))


def score_raised_arms_pose(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    wrists_above_head = (
        landmarks[LEFT_WRIST]["y"] < landmarks[NOSE]["y"]
        and landmarks[RIGHT_WRIST]["y"] < landmarks[NOSE]["y"]
    )
    penalties = [
        linear_penalty(abs(shoulder_center["x"] - hip_center["x"]), 0.08, 0.28),
        angular_penalty(angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 170, 18, 0.16),
        angular_penalty(angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 170, 18, 0.16),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 175, 12, 0.10),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 175, 12, 0.10),
        0.20 if not wrists_above_head else 0.0,
    ]
    return pack("Raised-Arms", 1 - sum(penalties))


def score_forward_fold_pose(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    ankle_center = midpoint(landmarks, LEFT_ANKLE, RIGHT_ANKLE)
    wrists_to_ankles = (
        distance(landmarks[LEFT_WRIST], landmarks[LEFT_ANKLE])
        + distance(landmarks[RIGHT_WRIST], landmarks[RIGHT_ANKLE])
    ) / 2
    penalties = [
        0.0 if hip_center["y"] < shoulder_center["y"] else 0.26,
        linear_penalty(abs(shoulder_center["x"] - ankle_center["x"]), 0.18, 0.14),
        linear_penalty(wrists_to_ankles, 0.28, 0.28),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 160, 20, 0.12),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 160, 20, 0.12),
    ]
    return pack("Standing-Forward-Fold", 1 - sum(penalties))


def score_low_lunge_pose(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    left_knee = angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_knee = angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    front_knee = min(left_knee, right_knee)
    back_knee = max(left_knee, right_knee)
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)

    penalties = [
        angular_penalty(front_knee, 95, 20, 0.28),
        angular_penalty(back_knee, 160, 25, 0.18),
        linear_penalty(abs(shoulder_center["x"] - hip_center["x"]), 0.14, 0.18),
    ]
    return pack("Low-Lunge", 1 - sum(penalties))


def score_plank_pose(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    ankle_center = midpoint(landmarks, LEFT_ANKLE, RIGHT_ANKLE)
    penalties = [
        linear_penalty(abs(shoulder_center["y"] - hip_center["y"]), 0.08, 0.22),
        linear_penalty(abs(hip_center["y"] - ankle_center["y"]), 0.08, 0.22),
        angular_penalty(angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 170, 18, 0.14),
        angular_penalty(angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 170, 18, 0.14),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 170, 18, 0.14),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 170, 18, 0.14),
    ]
    return pack("Plank", 1 - sum(penalties))


def score_cobra_pose(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    penalties = [
        0.0 if shoulder_center["y"] < hip_center["y"] else 0.28,
        angular_penalty(angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 150, 35, 0.16),
        angular_penalty(angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 150, 35, 0.16),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 170, 18, 0.12),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 170, 18, 0.12),
    ]
    return pack("Cobra", 1 - sum(penalties))


def score_downward_dog_pose(landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    penalties = [
        0.0 if hip_center["y"] < shoulder_center["y"] else 0.30,
        angular_penalty(angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 170, 18, 0.14),
        angular_penalty(angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 170, 18, 0.14),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 165, 20, 0.14),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 165, 20, 0.14),
    ]
    return pack("Downward-Dog", 1 - sum(penalties))


def pack(label: str, confidence: float) -> dict[str, Any]:
    return {"label": label, "confidence": max(0.0, min(0.99, round(confidence, 4)))}


def midpoint(landmarks: list[dict[str, Any]], left_index: int, right_index: int) -> dict[str, float]:
    return {
        "x": (landmarks[left_index]["x"] + landmarks[right_index]["x"]) / 2,
        "y": (landmarks[left_index]["y"] + landmarks[right_index]["y"]) / 2,
        "z": (landmarks[left_index]["z"] + landmarks[right_index]["z"]) / 2,
    }


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def angle(landmarks: list[dict[str, Any]], first: int, middle: int, last: int) -> float:
    a = landmarks[first]
    b = landmarks[middle]
    c = landmarks[last]
    ba = (a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"])
    bc = (c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"])
    numerator = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    denominator = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2) * math.sqrt(
        bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2
    )
    if denominator == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, numerator / denominator))
    return math.degrees(math.acos(cosine))


def angular_penalty(actual: float, target: float, tolerance: float, max_penalty: float) -> float:
    excess = max(0.0, abs(actual - target) - tolerance)
    return min(max_penalty, (excess / 45.0) * max_penalty)


def linear_penalty(actual: float, tolerance: float, max_penalty: float) -> float:
    excess = max(0.0, actual - tolerance)
    return min(max_penalty, (excess / max(tolerance, 0.01)) * max_penalty)
