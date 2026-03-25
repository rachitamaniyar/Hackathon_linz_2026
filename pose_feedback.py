import math
from typing import Any


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
LEFT_HEEL = 29
RIGHT_HEEL = 30


def generate_pose_feedback(label: str | None, landmarks: list[dict[str, Any]]) -> dict[str, Any]:
    if not label or not landmarks:
        return {
            "pose_score": 0,
            "summary": "No confident pose prediction was available.",
            "suggestions": [
                "Retake the pose with your full body visible in the frame.",
                "Use good lighting and keep the camera far enough to include the whole pose.",
            ],
        }

    evaluators = {
        "Prayer": evaluate_prayer,
        "Raised-Arms": evaluate_raised_arms,
        "Standing-Forward-Fold": evaluate_forward_fold,
        "Low-Lunge": evaluate_low_lunge,
        "Plank": evaluate_plank,
        "Standing-Mountain": evaluate_mountain,
        "Tree": evaluate_tree,
        "Warrior": evaluate_warrior,
        "Triangle": evaluate_triangle,
        "Downward-Dog": evaluate_downward_dog,
        "Cobra": evaluate_cobra,
        "Child": evaluate_child,
        "Bridge": evaluate_bridge,
        "Pigeon": evaluate_pigeon,
    }

    evaluator = evaluators.get(label, evaluate_generic)
    return evaluator(landmarks, label)


def evaluate_generic(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    shoulder_tilt = linear_penalty(level_gap(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER), 0.08, 12)
    hip_tilt = linear_penalty(level_gap(landmarks, LEFT_HIP, RIGHT_HIP), 0.08, 12)
    torso_shift = linear_penalty(torso_offset(landmarks), 0.10, 20)
    score = clamp_score(100 - shoulder_tilt - hip_tilt - torso_shift)
    suggestions = []

    if shoulder_tilt:
        suggestions.append("Level your shoulders before the next capture.")
    if hip_tilt:
        suggestions.append("Stabilize your hips and distribute weight evenly.")
    if torso_shift:
        suggestions.append("Keep your torso centered and avoid leaning sideways.")
    if not suggestions:
        suggestions.append("Pose looks stable in this frame.")

    return {
        "pose_score": score,
        "summary": f"{label} analyzed with general alignment checks.",
        "suggestions": suggestions[:4],
    }


def evaluate_mountain(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    penalties = [
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 175, 10, 18),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 175, 10, 18),
        linear_penalty(level_gap(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER), 0.08, 12),
        linear_penalty(level_gap(landmarks, LEFT_HIP, RIGHT_HIP), 0.08, 12),
        linear_penalty(torso_offset(landmarks), 0.08, 18),
    ]
    suggestions = []

    if penalties[0] or penalties[1]:
        suggestions.append("Straighten both legs and engage your knees.")
    if penalties[2]:
        suggestions.append("Keep your shoulders level and relaxed.")
    if penalties[3]:
        suggestions.append("Square your hips to the camera.")
    if penalties[4]:
        suggestions.append("Stack your shoulders over your hips.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Mountain pose checks completed.")


def evaluate_prayer(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    wrist_center = midpoint(landmarks, LEFT_WRIST, RIGHT_WRIST)
    wrist_gap = distance(landmarks[LEFT_WRIST], landmarks[RIGHT_WRIST])
    penalties = [
        linear_penalty(abs(shoulder_center["x"] - hip_center["x"]), 0.06, 16),
        linear_penalty(wrist_gap, 0.10, 22),
        linear_penalty(abs(wrist_center["y"] - ((shoulder_center["y"] + hip_center["y"]) / 2)), 0.16, 18),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 175, 12, 10),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 175, 12, 10),
    ]
    suggestions = []

    if penalties[0]:
        suggestions.append("Stand tall with shoulders stacked over hips.")
    if penalties[1]:
        suggestions.append("Bring the palms closer together at the chest.")
    if penalties[2]:
        suggestions.append("Keep the prayer hands centered at heart level.")
    if penalties[3] or penalties[4]:
        suggestions.append("Keep both legs straight and grounded.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Prayer pose checks completed.")


def evaluate_raised_arms(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    wrists_above_head = (
        landmarks[LEFT_WRIST]["y"] < landmarks[LEFT_SHOULDER]["y"]
        and landmarks[RIGHT_WRIST]["y"] < landmarks[RIGHT_SHOULDER]["y"]
    )
    penalties = [
        linear_penalty(abs(shoulder_center["x"] - hip_center["x"]), 0.08, 16),
        angular_penalty(angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 170, 20, 16),
        angular_penalty(angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 170, 20, 16),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 175, 12, 10),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 175, 12, 10),
        18 if not wrists_above_head else 0,
    ]
    suggestions = []

    if penalties[0]:
        suggestions.append("Lift through the torso and avoid leaning sideways.")
    if penalties[1] or penalties[2]:
        suggestions.append("Straighten both arms overhead.")
    if penalties[3] or penalties[4]:
        suggestions.append("Keep both legs long and steady.")
    if penalties[5]:
        suggestions.append("Reach the hands higher above the head.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Raised arms checks completed.")


def evaluate_forward_fold(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    wrist_to_ankle = average_distance(landmarks, (LEFT_WRIST, LEFT_ANKLE), (RIGHT_WRIST, RIGHT_ANKLE))
    penalties = [
        18 if hip_center["y"] >= shoulder_center["y"] else 0,
        linear_penalty(wrist_to_ankle, 0.28, 24),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 160, 20, 12),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 160, 20, 12),
        linear_penalty(abs(shoulder_center["x"] - midpoint(landmarks, LEFT_ANKLE, RIGHT_ANKLE)["x"]), 0.18, 14),
    ]
    suggestions = []

    if penalties[0]:
        suggestions.append("Hinge deeper from the hips so the torso folds over the legs.")
    if penalties[1]:
        suggestions.append("Reach the hands closer to the ankles or floor.")
    if penalties[2] or penalties[3]:
        suggestions.append("Lengthen the backs of the legs.")
    if penalties[4]:
        suggestions.append("Keep the torso centered over the feet.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Forward fold checks completed.")


def evaluate_low_lunge(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    left_knee = angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_knee = angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    front_knee = min(left_knee, right_knee)
    back_knee = max(left_knee, right_knee)
    penalties = [
        angular_penalty(front_knee, 95, 20, 24),
        angular_penalty(back_knee, 160, 25, 18),
        linear_penalty(torso_offset(landmarks), 0.14, 16),
    ]
    suggestions = []

    if penalties[0]:
        suggestions.append("Bend the front knee more and stack it over the ankle.")
    if penalties[1]:
        suggestions.append("Extend the back leg farther behind you.")
    if penalties[2]:
        suggestions.append("Lift the chest and keep the torso centered.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Low lunge checks completed.")


def evaluate_plank(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    ankle_center = midpoint(landmarks, LEFT_ANKLE, RIGHT_ANKLE)
    penalties = [
        linear_penalty(abs(shoulder_center["y"] - hip_center["y"]), 0.08, 18),
        linear_penalty(abs(hip_center["y"] - ankle_center["y"]), 0.08, 18),
        angular_penalty(angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 170, 18, 14),
        angular_penalty(angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 170, 18, 14),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 170, 18, 14),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 170, 18, 14),
    ]
    suggestions = []

    if penalties[0] or penalties[1]:
        suggestions.append("Keep the body in one straight line from shoulders to ankles.")
    if penalties[2] or penalties[3]:
        suggestions.append("Press the floor away and keep the arms straight.")
    if penalties[4] or penalties[5]:
        suggestions.append("Engage the legs and avoid sagging through the hips.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Plank checks completed.")


def evaluate_tree(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    left_knee = angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_knee = angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    standing_side = "left" if left_knee >= right_knee else "right"
    standing_knee = left_knee if standing_side == "left" else right_knee
    bent_knee = right_knee if standing_side == "left" else left_knee
    standing_knee_idx = LEFT_KNEE if standing_side == "left" else RIGHT_KNEE
    bent_ankle_idx = RIGHT_ANKLE if standing_side == "left" else LEFT_ANKLE
    foot_lift = landmarks[bent_ankle_idx]["y"] < landmarks[standing_knee_idx]["y"]

    penalties = [
        angular_penalty(standing_knee, 175, 10, 20),
        angular_penalty(bent_knee, 70, 35, 18),
        linear_penalty(torso_offset(landmarks), 0.10, 18),
        0 if foot_lift else 16,
    ]
    suggestions = []

    if penalties[0]:
        suggestions.append("Lock in the standing leg and keep it straight.")
    if penalties[1]:
        suggestions.append("Open the bent knee outward more.")
    if penalties[2]:
        suggestions.append("Keep your torso upright and centered.")
    if penalties[3]:
        suggestions.append("Lift the bent foot higher on the standing leg.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Tree pose checks completed.")


def evaluate_warrior(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    left_knee = angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_knee = angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    front_knee = left_knee if left_knee <= right_knee else right_knee
    back_knee = right_knee if left_knee <= right_knee else left_knee

    penalties = [
        angular_penalty(front_knee, 100, 18, 22),
        angular_penalty(back_knee, 175, 10, 18),
        linear_penalty(abs(landmarks[LEFT_WRIST]["y"] - landmarks[LEFT_SHOULDER]["y"]), 0.10, 12),
        linear_penalty(abs(landmarks[RIGHT_WRIST]["y"] - landmarks[RIGHT_SHOULDER]["y"]), 0.10, 12),
        linear_penalty(torso_offset(landmarks), 0.10, 14),
    ]
    suggestions = []

    if penalties[0]:
        suggestions.append("Bend the front knee more so it approaches a right angle.")
    if penalties[1]:
        suggestions.append("Straighten the back leg.")
    if penalties[2] or penalties[3]:
        suggestions.append("Keep both arms extended at shoulder height.")
    if penalties[4]:
        suggestions.append("Keep your torso upright over the hips.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Warrior pose checks completed.")


def evaluate_triangle(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    wrist_span = abs(landmarks[LEFT_WRIST]["y"] - landmarks[RIGHT_WRIST]["y"])
    torso_side_bend = abs(midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)["x"] - midpoint(landmarks, LEFT_HIP, RIGHT_HIP)["x"])
    penalties = [
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 175, 10, 18),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 175, 10, 18),
        linear_penalty(torso_side_bend, 0.15, 20),
        linear_penalty(wrist_span, 0.30, 12),
    ]
    suggestions = []

    if penalties[0] or penalties[1]:
        suggestions.append("Keep both legs straight and strong.")
    if penalties[2]:
        suggestions.append("Lengthen the torso more to the side before reaching down.")
    if penalties[3]:
        suggestions.append("Reach one arm up and the other down in one vertical line.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Triangle pose checks completed.")


def evaluate_downward_dog(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    ankle_center = midpoint(landmarks, LEFT_ANKLE, RIGHT_ANKLE)
    penalties = [
        angular_penalty(angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 170, 12, 14),
        angular_penalty(angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 170, 12, 14),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 170, 15, 14),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 170, 15, 14),
        0 if hip_center["y"] < shoulder_center["y"] else 22,
        0 if hip_center["y"] < ankle_center["y"] else 10,
    ]
    suggestions = []

    if penalties[0] or penalties[1]:
        suggestions.append("Press through the hands and straighten both arms.")
    if penalties[2] or penalties[3]:
        suggestions.append("Lengthen the backs of the legs.")
    if penalties[4] or penalties[5]:
        suggestions.append("Lift the hips higher to form an inverted V shape.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Downward Dog checks completed.")


def evaluate_cobra(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    penalties = [
        angular_penalty(angle(landmarks, LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 160, 25, 12),
        angular_penalty(angle(landmarks, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST), 160, 25, 12),
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 175, 15, 10),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 175, 15, 10),
        0 if shoulder_center["y"] < hip_center["y"] else 22,
    ]
    suggestions = []

    if penalties[0] or penalties[1]:
        suggestions.append("Open the chest and press a little more through the arms.")
    if penalties[2] or penalties[3]:
        suggestions.append("Keep both legs extended behind you.")
    if penalties[4]:
        suggestions.append("Lift the chest higher than the hips.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Cobra pose checks completed.")


def evaluate_child(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    penalties = [
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 70, 30, 16),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 70, 30, 16),
        linear_penalty(average_distance(landmarks, (LEFT_HIP, LEFT_HEEL), (RIGHT_HIP, RIGHT_HEEL)), 0.22, 18),
        linear_penalty(average_distance(landmarks, (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)), 0.25, 18),
    ]
    suggestions = []

    if penalties[0] or penalties[1]:
        suggestions.append("Fold the knees more deeply under the hips.")
    if penalties[2]:
        suggestions.append("Sink the hips back toward the heels.")
    if penalties[3]:
        suggestions.append("Lower the chest closer to the floor.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Child pose checks completed.")


def evaluate_bridge(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    hip_center = midpoint(landmarks, LEFT_HIP, RIGHT_HIP)
    shoulder_center = midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)
    penalties = [
        angular_penalty(angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), 100, 25, 16),
        angular_penalty(angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE), 100, 25, 16),
        0 if hip_center["y"] < shoulder_center["y"] else 24,
    ]
    suggestions = []

    if penalties[0] or penalties[1]:
        suggestions.append("Bring the knees closer to a 90 degree bend.")
    if penalties[2]:
        suggestions.append("Lift the hips higher through the center of the body.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Bridge pose checks completed.")


def evaluate_pigeon(landmarks: list[dict[str, Any]], label: str) -> dict[str, Any]:
    penalties = [
        linear_penalty(torso_offset(landmarks), 0.12, 18),
        linear_penalty(level_gap(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER), 0.10, 12),
    ]
    suggestions = []

    if penalties[0]:
        suggestions.append("Lift the chest and keep the torso centered over the hips.")
    if penalties[1]:
        suggestions.append("Keep the shoulders even and relaxed.")
    if not suggestions:
        suggestions.append("Pose looks stable in this captured frame.")

    return pack_feedback(label, clamp_score(100 - sum(penalties)), suggestions, "Pigeon pose checks completed.")


def pack_feedback(label: str, score: int, suggestions: list[str], summary: str) -> dict[str, Any]:
    return {
        "pose_score": score,
        "summary": f"{label}: {summary}",
        "suggestions": suggestions[:4] if suggestions else ["Pose looks stable in this captured frame."],
    }


def midpoint(landmarks: list[dict[str, Any]], left_index: int, right_index: int) -> dict[str, float]:
    return {
        "x": (landmarks[left_index]["x"] + landmarks[right_index]["x"]) / 2,
        "y": (landmarks[left_index]["y"] + landmarks[right_index]["y"]) / 2,
        "z": (landmarks[left_index]["z"] + landmarks[right_index]["z"]) / 2,
    }


def distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def average_distance(landmarks: list[dict[str, Any]], pair_a: tuple[int, int], pair_b: tuple[int, int]) -> float:
    return (
        distance(landmarks[pair_a[0]], landmarks[pair_a[1]])
        + distance(landmarks[pair_b[0]], landmarks[pair_b[1]])
    ) / 2


def torso_offset(landmarks: list[dict[str, Any]]) -> float:
    return abs(midpoint(landmarks, LEFT_SHOULDER, RIGHT_SHOULDER)["x"] - midpoint(landmarks, LEFT_HIP, RIGHT_HIP)["x"])


def level_gap(landmarks: list[dict[str, Any]], left_index: int, right_index: int) -> float:
    return abs(landmarks[left_index]["y"] - landmarks[right_index]["y"])


def angle(landmarks: list[dict[str, Any]], first: int, middle: int, last: int) -> float:
    a = landmarks[first]
    b = landmarks[middle]
    c = landmarks[last]
    ba = (a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"])
    bc = (c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"])
    numerator = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    denominator = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)
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


def clamp_score(score: float) -> int:
    return max(0, min(100, int(round(score))))
