"""Long-term memory by fixed rules: pull facts about the user out of their own messages.

No model is involved (a 57M model cannot summarize reliably). Only the user's messages are read,
never the model's replies, so the model cannot write its own hallucinations into memory.

Kinds marked single-valued replace the previous value ("tôi 22 tuổi" then "tôi 23 tuổi" keeps 23).
"""

import re
from dataclasses import dataclass

ME = r"(?:tôi|mình|tớ|em)"
STOP = r"(?=\s+(?:và|nhưng|còn|nên|vì)\s|[.,;!?\n]|$)"
VALUE = r"([^.,;!?\n]{1,40}?)"

QUESTION_WORDS = re.compile(r"\b(?:gì|đâu|nào|bao nhiêu|mấy|sao|không nhỉ|chưa nhỉ)\b", re.IGNORECASE)

SINGLE_VALUED = {"name", "age", "location", "job"}
LABELS = {"name": "Tên", "age": "Tuổi", "location": "Sống ở", "job": "Nghề", "likes": "Thích", "note": "Ghi nhớ"}

RULES: list[tuple[str, re.Pattern]] = [
    ("note", re.compile(r"^\s*(?:hãy\s+)?(?:ghi nhớ|nhớ giúp(?:\s+(?:tôi|mình))?|nhớ(?=\s+(?:là|rằng)\s))(?:\s+(?:là|rằng))?\s*[:,]?\s*(.{2,200}?)\s*$",
                        re.IGNORECASE)),
    ("name", re.compile(rf"(?:tên (?:của )?{ME} là|{ME} tên(?: là)?)\s+{VALUE}{STOP}", re.IGNORECASE)),
    ("age", re.compile(rf"{ME} (?:năm nay\s+)?(\d{{1,3}})\s+tuổi", re.IGNORECASE)),
    ("location", re.compile(rf"(?:{ME}|\bvà) (?:đang\s+)?(?:sống (?:ở|tại)|đến từ|quê (?:ở\s+)?)\s*{VALUE}{STOP}",
                            re.IGNORECASE)),
    ("job", re.compile(rf"{ME} (?:làm nghề|là|đang là|đang làm)\s+((?:sinh viên|học sinh|lập trình viên|kỹ sư|giáo viên|"
                       rf"bác sĩ|y tá|nhân viên|kế toán|designer|developer|giảng viên)[^.,;!?\n]{{0,30}}?){STOP}",
                       re.IGNORECASE)),
    ("likes", re.compile(rf"{ME} (?:rất\s+)?(?:thích|yêu|mê)\s+{VALUE}{STOP}", re.IGNORECASE)),
]


@dataclass(frozen=True)
class Fact:
    kind: str
    value: str

    @property
    def text(self) -> str:
        return f"{LABELS[self.kind]}: {self.value}"


RECALL_RULES: list[tuple[str | None, re.Pattern]] = [
    ("name", re.compile(rf"(?:{ME} tên(?: là)? gì|tên (?:của )?{ME} là gì)", re.IGNORECASE)),
    ("age", re.compile(rf"{ME} (?:bao nhiêu|mấy) tuổi", re.IGNORECASE)),
    ("location", re.compile(rf"{ME} (?:sống|ở) (?:ở |tại )?đâu|quê {ME} ở đâu|{ME} đến từ đâu", re.IGNORECASE)),
    ("job", re.compile(rf"{ME} làm (?:nghề )?gì", re.IGNORECASE)),
    ("likes", re.compile(rf"{ME} thích (?:gì|cái gì|những gì)", re.IGNORECASE)),
    (None, re.compile(rf"(?:bạn )?(?:nhớ|biết) (?:gì|những gì) về {ME}", re.IGNORECASE)),
]


def recall_answer(message: str, memories: list[dict]) -> str | None:
    """Answer "what's my name / how old am I / what do you remember about me" straight from memory.

    The 57M model reads memories in the system turn only some of the time, so these questions skip it.
    Returns None when the question is not a recall question or nothing relevant is stored.
    """
    for kind, pattern in RECALL_RULES:
        if not pattern.search(message):
            continue
        relevant = [m for m in memories if kind is None or m["kind"] == kind]
        if not relevant:
            return None
        return "Mình nhớ: " + "; ".join(f"{LABELS[m['kind']].lower()} {m['value']}" for m in relevant) + "."
    return None


def extract_facts(message: str) -> list[Fact]:
    """Facts stated in one user message. Questions ("tôi tên gì?") yield nothing."""
    note = RULES[0][1].match(message)
    if note:
        return [Fact("note", note.group(1).strip())]
    # Questions often come without "?": "tên mình là gì vậy bạn" must not store the name "gì vậy bạn".
    if message.rstrip().endswith("?") or QUESTION_WORDS.search(message):
        return []
    facts = []
    for kind, pattern in RULES[1:]:
        match = pattern.search(message)
        if match and match.group(1).strip():
            facts.append(Fact(kind, match.group(1).strip()))
    return facts
