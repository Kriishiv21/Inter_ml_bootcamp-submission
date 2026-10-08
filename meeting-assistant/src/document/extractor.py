from src.schemas import MeetingRecord, ActionItem, KeyDecision, EvidenceVerification
from src.llm_client import call_llm_structured
from src.config import DOCUMENTER_MODEL

# --- 1. FORMAT FOR TRACEABILITY ---
def format_transcript_with_ids(segments: list[dict]) -> str:
    """Formats the transcript so the LLM sees the U-IDs for every line."""
    formatted = []
    for seg in segments:
        uid = seg.get("id", "U???")
        text = seg.get("text", "").strip()
        formatted.append(f"[{uid}] {text}")
    return "\n".join(formatted)

# --- 2. EXTRACTION (PHASE 3) ---
def extract_meeting_record(transcript_with_ids: str) -> MeetingRecord:
    """Uses LLM 2 to extract the structured MOM and tasks with citations."""
    system_prompt = (
        "You are a strict meeting minutes generator. "
        "Read the provided transcript where each line starts with an Utterance ID (e.g., [U001]). "
        "Extract a concise summary, key decisions, and action items. "
        "CRITICAL RULES: "
        "1. Every decision and action item MUST cite the specific Utterance IDs that prove it. "
        "2. For action items, if an owner is not explicitly stated in the text by name, you MUST set owner to 'Unassigned'. "
        "3. If a deadline is not explicitly stated, you MUST set deadline to 'Unspecified'. "
        "4. Do not include unconfirmed proposals in the decisions list."
    )
    print("Extracting Meeting Record...")
    return call_llm_structured(transcript_with_ids, system_prompt, DOCUMENTER_MODEL, MeetingRecord)

# --- 3. THE QUALITY GATE (PHASE 4) ---

def check_evidence_support(
    claim: str,
    evidence_ids: list[str],
    segments: list[dict]
) -> EvidenceVerification:
    """Checks whether the cited utterances actually support the claim."""

    segment_by_id = {
        seg.get("id"): seg.get("text", "").strip()
        for seg in segments
    }

    evidence_text = "\n".join(
        f"[{uid}] {segment_by_id[uid]}"
        for uid in evidence_ids
        if uid in segment_by_id
    )

    system_prompt = (
        "You are a strict evidence validator for a meeting assistant. "
        "Determine whether the cited meeting utterances actually support "
        "the generated claim.\n\n"
        "Rules:\n"
        "1. Use only the provided utterances.\n"
        "2. The cited evidence must directly support the claim.\n"
        "3. Do not infer unstated owners or deadlines.\n"
        "4. Do not turn a proposal into an agreed decision.\n"
        "5. Do not turn uncertain language into a confirmed commitment.\n"
        "6. If any cited utterance does not contribute to supporting the claim, "
        "list its U-ID as unsupported.\n"
    )

    prompt = f"""
CLAIM:
{claim}

CITED UTTERANCES:
{evidence_text}

Determine whether the cited utterances support the claim.
"""

    return call_llm_structured(
        prompt,
        system_prompt,
        DOCUMENTER_MODEL,
        EvidenceVerification
    )


def verify_and_filter_record(
    record: MeetingRecord,
    segments: list[dict]
) -> MeetingRecord:
    """Drops decisions and tasks whose cited evidence does not support the claim."""

    print("Running Evidence Grounding Quality Gate...")

    valid_uids = {
        seg.get("id")
        for seg in segments
        if seg.get("id")
    }

    verified_decisions = []

    for dec in record.decisions:
        valid_cites = [
            uid for uid in dec.evidence_utterance_ids
            if uid in valid_uids
        ]

        if not valid_cites:
            print(f"DROPPED DECISION (No valid evidence): {dec.decision}")
            continue

        verification = check_evidence_support(
            claim=dec.decision,
            evidence_ids=valid_cites,
            segments=segments
        )

        if verification.is_supported:
            dec.evidence_utterance_ids = valid_cites
            verified_decisions.append(dec)
        else:
            print(
                f"DROPPED DECISION (Unsupported evidence): "
                f"{dec.decision}"
            )
            print(f"-> {verification.reason}")

    verified_tasks = []

    for task in record.action_items:
        valid_cites = [
            uid for uid in task.evidence_utterance_ids
            if uid in valid_uids
        ]

        if not valid_cites:
            print(f"DROPPED TASK (No valid evidence): {task.description}")
            continue

        claim = (
            f"Task: {task.description} | "
            f"Owner: {task.owner} | "
            f"Deadline: {task.deadline}"
        )

        verification = check_evidence_support(
            claim=claim,
            evidence_ids=valid_cites,
            segments=segments
        )

        if verification.is_supported:
            task.evidence_utterance_ids = valid_cites
            verified_tasks.append(task)
        else:
            print(
                f"DROPPED TASK (Unsupported evidence): "
                f"{task.description}"
            )
            print(f"-> {verification.reason}")

    return MeetingRecord(
        summary=record.summary,
        decisions=verified_decisions,
        action_items=verified_tasks
    )

# --- 4. UNIFIED RENDERING ---
def generate_markdown(record: MeetingRecord) -> str:
    """Renders the Markdown directly from the verified JSON record."""
    md = f"# Meeting Summary\n\n{record.summary}\n\n"
    
    md += "## Key Decisions\n"
    if not record.decisions:
        md += "* None stated.\n"
    for d in record.decisions:
        cites = ", ".join(d.evidence_utterance_ids)
        md += f"* {d.decision} *(Evidence: {cites})*\n"
        
    md += "\n## Action Items\n"
    if not record.action_items:
        md += "* None assigned.\n"
    for t in record.action_items:
        cites = ", ".join(t.evidence_utterance_ids)
        md += f"* **{t.description}** | Owner: {t.owner} | Due: {t.deadline} *(Evidence: {cites})*\n"
        
    return md