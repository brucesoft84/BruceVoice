"""MEDLATEC Vietnam call-center domain analysis.

Provides intent classification, entity extraction, and call-summary
generation for healthcare calls (test-result inquiries and medical
appointment scheduling — at home or at hospital).
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ── Intent keyword sets ────────────────────────────────────────────────────────

# Phrases strongly associated with asking about a test result
_INTENT_TEST_RESULT = [
    "kết quả", "kết quả xét nghiệm", "kết quả khám", "kết quả siêu âm",
    "kết quả chụp", "kết quả mri", "kết quả ct", "kết quả máu",
    "xem kết quả", "lấy kết quả", "nhận kết quả", "có kết quả chưa",
    "ra kết quả chưa", "kết quả như thế nào", "bao giờ có kết quả",
    "kết quả xét nghiệm máu", "kết quả nước tiểu",
]

# Phrases associated with scheduling at a clinic / hospital
_INTENT_SCHEDULE_HOSPITAL = [
    "đặt lịch khám", "đặt lịch tại bệnh viện", "đặt lịch tại cơ sở",
    "đặt lịch tại phòng khám", "đặt lịch khám tại", "đặt khám",
    "hẹn khám", "hẹn tại bệnh viện", "tới cơ sở", "đến cơ sở",
    "khám tại bệnh viện", "khám ngoại trú", "khám theo yêu cầu",
    "tại cơ sở medlatec", "tại phòng khám medlatec",
]

# Phrases associated with scheduling at-home sample collection
_INTENT_SCHEDULE_HOME = [
    "lấy mẫu tại nhà", "xét nghiệm tại nhà", "đặt lịch tại nhà",
    "lấy máu tại nhà", "lấy mẫu về nhà", "đến nhà lấy mẫu",
    "dịch vụ tại nhà", "lấy mẫu tận nhà", "lấy mẫu tận nơi",
    "lấy máu tại nhà", "cử người đến nhà", "đặt dịch vụ tại nhà",
    "mẫu tại nhà", "lấy mẫu máu tại nhà",
]

# General service / information inquiry
_INTENT_SERVICE_INQUIRY = [
    "dịch vụ", "giá dịch vụ", "chi phí xét nghiệm", "giá xét nghiệm",
    "xét nghiệm gì", "các loại xét nghiệm", "gói xét nghiệm",
    "xét nghiệm nào", "cần xét nghiệm", "quy trình", "thủ tục",
    "giờ làm việc", "địa chỉ medlatec", "cơ sở nào", "liên hệ",
]

# ── Entity extraction patterns ─────────────────────────────────────────────────

# Vietnamese phone number: 10 digits starting with 0 (or 84)
_RE_PHONE = re.compile(
    r"(?<!\d)"                         # no preceding digit
    r"(?:(?:\+84|84)[\s.-]?)?0?"       # optional country code
    r"(?:0[3-9]\d{8}|0[1-9]\d{7,8})"  # 10-digit VN mobile / landline
    r"(?!\d)"
)

# Time hints (Vietnamese — hours / dates / periods)
_RE_TIME = re.compile(
    r"(?:"
    r"(?:lúc|hồi|vào|khoảng|trước|sau)\s+\d{1,2}(?:[:h]\d{0,2})?\s*(?:giờ|h|am|pm)?"
    r"|"
    r"\d{1,2}[h:]\d{2}"
    r"|"
    r"(?:sáng|chiều|tối|trưa|ngày mai|hôm nay|hôm qua|thứ [2-7]|chủ nhật|cuối tuần)"
    r")"
)

# Common medical tests mentioned in MEDLATEC calls
_MEDICAL_TESTS = [
    "xét nghiệm máu", "công thức máu", "sinh hóa máu", "nước tiểu",
    "siêu âm", "chụp x-quang", "chụp mri", "chụp ct",
    "điện tim", "đo huyết áp", "xét nghiệm gan", "xét nghiệm thận",
    "xét nghiệm tiểu đường", "đường huyết", "cholesterol", "mỡ máu",
    "xét nghiệm tuyến giáp", "hormone", "ung thư", "psa",
    "xét nghiệm covid", "covid", "viêm gan", "hiv",
    "xét nghiệm sàng lọc", "sàng lọc ung thư",
]

# Districts / areas commonly served by MEDLATEC Hanoi
_AREAS = [
    "hà nội", "hoàn kiếm", "đống đa", "ba đình", "hai bà trưng",
    "hoàng mai", "thanh xuân", "cầu giấy", "tây hồ", "long biên",
    "gia lâm", "đông anh", "sóc sơn", "mê linh", "từ liêm",
    "nam từ liêm", "bắc từ liêm", "hà đông", "sơn tây",
    "hồ chí minh", "bình thạnh", "tân bình", "gò vấp", "thủ đức",
]


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_intent(text: str) -> str:
    """Classify the primary intent of the customer in the call.

    Scans the lowercased full transcript (or customer utterances) for
    domain-specific keyword patterns.

    Args:
        text: Combined text to scan (full transcript or customer text).

    Returns:
        One of: ``"ask_test_result"``, ``"schedule_home"``,
        ``"schedule_hospital"``, ``"service_inquiry"``, ``"other"``.
    """
    t = text.lower()

    scores = {
        "ask_test_result": sum(1 for kw in _INTENT_TEST_RESULT if kw in t),
        "schedule_home": sum(1 for kw in _INTENT_SCHEDULE_HOME if kw in t),
        "schedule_hospital": sum(1 for kw in _INTENT_SCHEDULE_HOSPITAL if kw in t),
        "service_inquiry": sum(1 for kw in _INTENT_SERVICE_INQUIRY if kw in t),
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "other"
    return best


def extract_entities(text: str) -> dict:
    """Extract named entities relevant to MEDLATEC calls.

    Args:
        text: Full transcript text.

    Returns:
        Dict with keys:
        - ``phones`` (list[str]): detected phone numbers.
        - ``tests`` (list[str]): mentioned medical tests.
        - ``times`` (list[str]): time / date mentions.
        - ``areas`` (list[str]): geographic areas mentioned.
        - ``location_type`` (str | None): ``"home"`` or ``"hospital"`` or None.
    """
    t = text.lower()

    phones = list({m.group() for m in _RE_PHONE.finditer(text)})
    times = list({m.group().strip() for m in _RE_TIME.finditer(t)})
    tests = [test for test in _MEDICAL_TESTS if test in t]
    areas = [area for area in _AREAS if area in t]

    # Location type heuristic
    home_signals = sum(1 for kw in _INTENT_SCHEDULE_HOME if kw in t)
    hosp_signals = sum(1 for kw in _INTENT_SCHEDULE_HOSPITAL if kw in t)
    if home_signals > hosp_signals:
        location_type = "home"
    elif hosp_signals > home_signals:
        location_type = "hospital"
    else:
        location_type = None

    return {
        "phones": phones,
        "tests": tests,
        "times": times,
        "areas": areas,
        "location_type": location_type,
    }


def generate_call_summary(
    conversation: list,
    transcript: str,
    roles: dict,
    duration: float,
) -> dict:
    """Generate a structured call summary for a MEDLATEC call.

    Args:
        conversation: Output of ``build_conversation_timeline()`` — list of
            utterance dicts ``{speaker, role, start, end, text, ...}``.
        transcript: Full merged transcript string (used as fallback text).
        roles: ``{speaker_id: "agent"|"customer"}`` dict.
        duration: Total audio duration in seconds.

    Returns:
        Dict with keys:
        - ``intent`` (str): classified call intent.
        - ``entities`` (dict): extracted entities.
        - ``turn_count`` (int): total conversational turns.
        - ``agent_turns`` (int): turns taken by the agent.
        - ``customer_turns`` (int): turns taken by the customer.
        - ``high_tone_turns`` (int): utterances flagged as high-tone.
        - ``duration_s`` (float): call duration in seconds.
        - ``outcome_hint`` (str): human-readable outcome guess.
    """
    # Build per-role text blocks
    customer_text = " ".join(
        u["text"] for u in conversation if u.get("role") == "customer"
    )
    agent_text = " ".join(
        u["text"] for u in conversation if u.get("role") == "agent"
    )
    # Use full transcript as fallback for intent / entity search
    search_text = customer_text or transcript

    intent = classify_intent(search_text)
    entities = extract_entities(transcript)

    turn_count = len(conversation)
    agent_turns = sum(1 for u in conversation if u.get("role") == "agent")
    customer_turns = sum(1 for u in conversation if u.get("role") == "customer")
    high_tone_turns = sum(1 for u in conversation if u.get("high_tone"))

    outcome_hint = _guess_outcome(intent, agent_text, entities)

    summary = {
        "intent": intent,
        "entities": entities,
        "turn_count": turn_count,
        "agent_turns": agent_turns,
        "customer_turns": customer_turns,
        "high_tone_turns": high_tone_turns,
        "duration_s": round(duration, 1),
        "outcome_hint": outcome_hint,
    }
    logger.info(
        "Call summary — intent: %s, entities: %s, outcome: %s",
        intent, entities, outcome_hint,
    )
    return summary


# ── Internal helpers ───────────────────────────────────────────────────────────

_OUTCOME_CONFIRMED = [
    "đã đặt", "đã ghi nhận", "đã tiếp nhận", "đã xác nhận", "đặt thành công",
    "đã ghi lại", "sẽ cử", "sẽ đến", "anh chị chờ", "nhân viên sẽ",
    "chúng tôi sẽ", "bên em sẽ", "xác nhận rồi", "ok rồi", "được rồi ạ",
]
_OUTCOME_PENDING = [
    "sẽ gọi lại", "gọi lại sau", "chờ xác nhận", "xem lại", "kiểm tra lại",
    "hỏi lại", "chờ bên em", "để em xem",
]
_OUTCOME_NO_RESULT = [
    "chưa có kết quả", "chưa ra kết quả", "đang xử lý", "đang chờ",
    "chưa xong", "kết quả chưa có", "chưa sẵn sàng",
]


def _guess_outcome(intent: str, agent_text: str, entities: dict) -> str:
    """Heuristically guess the call outcome from agent speech."""
    t = agent_text.lower()

    if intent == "ask_test_result":
        if any(kw in t for kw in _OUTCOME_NO_RESULT):
            return "result_not_ready"
        if any(kw in t for kw in _OUTCOME_CONFIRMED):
            return "result_provided"
        return "result_status_unclear"

    if intent in ("schedule_home", "schedule_hospital"):
        if any(kw in t for kw in _OUTCOME_CONFIRMED):
            loc = entities.get("location_type") or (
                "home" if intent == "schedule_home" else "hospital"
            )
            return f"appointment_confirmed_{loc}"
        if any(kw in t for kw in _OUTCOME_PENDING):
            return "appointment_pending"
        return "appointment_status_unclear"

    if intent == "service_inquiry":
        if any(kw in t for kw in _OUTCOME_CONFIRMED + _OUTCOME_PENDING):
            return "inquiry_handled"
        return "inquiry_status_unclear"

    return "unknown"
