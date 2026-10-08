import re
from src.schemas import TopicDictionary, CorrectionList, ContextualVerification, LocalTopicMap
from src.llm_client import call_llm_structured
from src.config import REFINER_MODEL

# --- 1. GLOBAL TOPIC PASS ---
def get_global_topic(raw_text: str) -> TopicDictionary:
    """Reads the raw transcript and extracts the main topic and a glossary of terms."""
    system_prompt = (
        "You are an expert meeting analyst. Read the following raw ASR transcript. "
        "Identify the primary domain topic. Then, extract a list of likely technical terms, "
        "acronyms, or proper nouns used in the meeting. Include terms that look like "
        "they might be phonetically misspelled ASR errors."
    )
    return call_llm_structured(raw_text, system_prompt, REFINER_MODEL, TopicDictionary)

# --- 2. LOCAL TOPIC PASS ---
def get_local_topics(
    transcript_with_ids: str,
    global_topic: str
) -> LocalTopicMap:
    """Groups consecutive utterances into local discussion topics."""

    system_prompt = (
        "You are an expert meeting analyst. "
        f"The overall meeting topic is: {global_topic}. "
        "Read the transcript with stable Utterance IDs. "
        "Divide the transcript into contiguous regions based on changes "
        "in the specific subject being discussed. "

        "For each region, provide the first and last Utterance ID, "
        "a short local topic, and relevant technical keywords. "

        "Rules: "
        "1. Keep regions reasonably broad; do not create a new topic for every sentence. "
        "2. Consecutive utterances discussing the same subject should belong to one region. "
        "3. Do not invent topics that are not supported by the transcript. "
        "4. Every region must use valid U-IDs from the transcript. "
        "5. Local topics should be concise, e.g. 'model training', "
        "'database migration', or 'deployment planning'."
    )

    return call_llm_structured(
        transcript_with_ids,
        system_prompt,
        REFINER_MODEL,
        LocalTopicMap
    )

# --- 3. SUSPICIOUS SPAN PROPOSAL ---
def propose_corrections(
    transcript_with_ids: str,
    topic_dict: TopicDictionary,
    local_topics: dict[str, dict]
) -> CorrectionList:
    """Finds localized ASR errors using global and local topic context."""

    topic_context = []

    for line in transcript_with_ids.splitlines():
        if not line.strip():
            continue

        uid = line.split("]", 1)[0].replace("[", "").strip()
        info = local_topics.get(
            uid,
            {"topic": topic_dict.global_topic, "keywords": []}
        )

        topic_context.append(
            f"{line}\n"
            f"  Local topic: {info['topic']}\n"
            f"  Local keywords: {', '.join(info['keywords'])}"
        )

    enriched_transcript = "\n".join(topic_context)

    system_prompt = (
        "You are a strict technical copy-editor correcting ASR errors. "
        "Each transcript line has a stable Utterance ID. "

        f"Global meeting topic: {topic_dict.global_topic}. "
        f"Global domain terms: {', '.join(topic_dict.keywords)}. "

        "Each utterance also has a local topic and local keywords. "
        "Use the local topic as the primary contextual signal for deciding "
        "whether an unusual word or phrase is plausible. "
        "Use the global topic as broader context. "

        "Identify plausible ASR errors, especially phonetic errors in "
        "technical terms, acronyms, proper nouns, and domain-specific language. "

        "For every proposed correction, you MUST provide the exact "
        "Utterance ID and exact raw span containing the error. "

        "DO NOT rewrite sentences. "
        "DO NOT fix grammar. "
        "DO NOT remove filler words. "
        "DO NOT change information that is not an ASR error. "
        "Do not correct a word merely because it is unusual or unfamiliar. "

        "Only output specific localized corrections."
    )

    return call_llm_structured(
        enriched_transcript,
        system_prompt,
        REFINER_MODEL,
        CorrectionList
    )

    return call_llm_structured(transcript_with_ids, system_prompt, REFINER_MODEL, CorrectionList)

def build_local_topic_lookup(
    local_topic_map: LocalTopicMap,
    segments: list[dict],
    global_topic: str
) -> dict[str, dict]:
    """Maps every U-ID to its local topic and keywords."""

    ordered_uids = [
        seg.get("id")
        for seg in segments
        if seg.get("id")
    ]

    uid_index = {
        uid: i
        for i, uid in enumerate(ordered_uids)
    }

    lookup = {}

    for region in local_topic_map.segments:
        start = uid_index.get(region.start_utterance_id)
        end = uid_index.get(region.end_utterance_id)

        if start is None or end is None or start > end:
            continue

        for uid in ordered_uids[start:end + 1]:
            lookup[uid] = {
                "topic": region.local_topic,
                "keywords": region.keywords
            }

    # Safe fallback for any U-ID not covered by the LLM.
    for uid in ordered_uids:
        if uid not in lookup:
            lookup[uid] = {
                "topic": global_topic,
                "keywords": []
            }

    return lookup

# --- 4. DETERMINISTIC GUARDRAILS ---

MONTHS = {
    "january": "01", "february": "02", "march": "03",
    "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09",
    "october": "10", "november": "11", "december": "12"
}

def extract_dates(text: str) -> list[str]:
    dates = []

    month_first = re.findall(
        r"\b(" + "|".join(MONTHS.keys()) +
        r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(\d{4}))?\b",
        text.lower()
    )

    for month, day, year in month_first:
        dates.append(
            f"{MONTHS[month]}-{int(day):02d}-{year or 'XXXX'}"
        )

    day_first = re.findall(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" +
        "|".join(MONTHS.keys()) +
        r")(?:\s+(\d{4}))?\b",
        text.lower()
    )

    for day, month, year in day_first:
        dates.append(
            f"{MONTHS[month]}-{int(day):02d}-{year or 'XXXX'}"
        )

    numeric_dates = re.findall(
        r"\b(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})\b",
        text
    )

    for a, b, c in numeric_dates:
        if len(a) == 4:
            dates.append(f"{int(b):02d}-{int(c):02d}-{a}")
        else:
            dates.append(f"{int(a):02d}-{int(b):02d}-{c}")

    return dates


def extract_times(text: str) -> list[str]:
    return re.findall(
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
        r"|\b\d{1,2}:\d{2}\b",
        text.lower()
    )


def extract_versions(text: str) -> list[str]:
    return re.findall(
        r"\bv?\d+(?:\.\d+)+(?:[-_][a-z0-9]+)?\b",
        text.lower()
    )


def extract_percentages(text: str) -> list[str]:
    return re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        text
    )


UNCERTAINTY_TERMS = {
    "maybe", "might", "possibly", "possibly",
    "perhaps", "could", "may", "probably",
    "likely", "unlikely", "uncertain",
    "not sure", "i think", "i believe"
}

COMMITMENT_TERMS = {
    "will", "must", "shall", "need to",
    "going to", "plan to", "agreed to",
    "commit", "committed", "ensure"
}


def extract_semantic_markers(
    text: str,
    markers: set[str]
) -> set[str]:
    text = text.lower()
    return {
        marker
        for marker in markers
        if re.search(
            rf"\b{re.escape(marker)}\b",
            text
        )
    }


def passes_deterministic_checks(
    raw_span: str,
    corrected_span: str
) -> bool:
    """Reject edits that alter protected information."""

    # 1. Number Preservation
    raw_nums = set(re.findall(r'\d+', raw_span))
    corr_nums = set(re.findall(r'\d+', corrected_span))

    if raw_nums != corr_nums:
        print(
            f"REJECTED: Number change detected "
            f"({raw_span} -> {corrected_span})"
        )
        return False

    # 2. Date Preservation
    raw_dates = extract_dates(raw_span)
    corr_dates = extract_dates(corrected_span)

    if raw_dates != corr_dates:
        print(
            f"REJECTED: Date change detected "
            f"({raw_span} -> {corrected_span})"
        )
        return False

    # 3. Time Preservation
    raw_times = extract_times(raw_span)
    corr_times = extract_times(corrected_span)

    if raw_times != corr_times:
        print(
            f"REJECTED: Time change detected "
            f"({raw_span} -> {corrected_span})"
        )
        return False

    # 4. Version Preservation
    raw_versions = extract_versions(raw_span)
    corr_versions = extract_versions(corrected_span)

    if raw_versions != corr_versions:
        print(
            f"REJECTED: Version change detected "
            f"({raw_span} -> {corrected_span})"
        )
        return False

    # 5. Percentage Preservation
    raw_percentages = extract_percentages(raw_span)
    corr_percentages = extract_percentages(corrected_span)

    if raw_percentages != corr_percentages:
        print(
            f"REJECTED: Percentage change detected "
            f"({raw_span} -> {corrected_span})"
        )
        return False

    # 6. Negation Preservation
    negations = r"\b(not|n't|never|no|none|cannot|won't)\b"

    raw_negs = set(
        re.findall(negations, raw_span.lower())
    )
    corr_negs = set(
        re.findall(negations, corrected_span.lower())
    )

    if raw_negs != corr_negs:
        print(
            f"REJECTED: Negation change detected "
            f"({raw_span} -> {corrected_span})"
        )
        return False

    # 7. Uncertainty Preservation
    raw_uncertainty = extract_semantic_markers(
        raw_span,
        UNCERTAINTY_TERMS
    )
    corr_uncertainty = extract_semantic_markers(
        corrected_span,
        UNCERTAINTY_TERMS
    )

    if raw_uncertainty != corr_uncertainty:
        print(
            f"REJECTED: Uncertainty change detected "
            f"({raw_span} -> {corrected_span})"
        )
        return False

    # 8. Commitment Preservation
    raw_commitment = extract_semantic_markers(
        raw_span,
        COMMITMENT_TERMS
    )
    corr_commitment = extract_semantic_markers(
        corrected_span,
        COMMITMENT_TERMS
    )

    if raw_commitment != corr_commitment:
        print(
            f"REJECTED: Commitment level change detected "
            f"({raw_span} -> {corrected_span})"
        )
        return False

    return True

# --- 5. CONTEXTUAL VERIFIER ---
def verify_context(proposed: str, local_context: str, topic_dict: TopicDictionary) -> bool:
    """Checks if the proposed term logically fits the domain context."""
    system_prompt = (
        "You are a verification gate. Does the proposed corrected term logically and "
        f"semantically fit within the meeting topic ({topic_dict.global_topic}) and the "
        "surrounding local sentence context? Return true if supported, false if hallucinated."
    )
    prompt = f"Context: {local_context}\nProposed Term: {proposed}"
    result = call_llm_structured(prompt, system_prompt, REFINER_MODEL, ContextualVerification)
    return result.is_supported if result else False

# --- 6. MASTER REFINER PIPELINE ---
def refine_transcript(segments: list[dict]) -> list[dict]:
    """Refines ASR segments while preserving their stable Utterance IDs."""
    # Build the full raw transcript for the global topic pass.
    raw_text = " ".join(
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("text", "").strip()
    )
    print("Step 1: Running Global Topic Pass...")
    topic_dict = get_global_topic(raw_text)

    if not topic_dict:
        return segments

    print(f"-> Topic: {topic_dict.global_topic}")
    # Build transcript with U-IDs for local topic analysis.
    transcript_with_ids = "\n".join(
        f"[{seg.get('id', 'U???')}] {seg.get('text', '').strip()}"
        for seg in segments
        if seg.get("text", "").strip()
    )

    print("Step 2: Running Local Topic Pass...")
    local_topic_map = get_local_topics(
        transcript_with_ids,
        topic_dict.global_topic
    )

    local_topics = build_local_topic_lookup(
        local_topic_map,
        segments,
        topic_dict.global_topic
    )

    print("Step 3: Proposing Corrections...")
    proposals = propose_corrections(
        transcript_with_ids,
        topic_dict,
        local_topics
    )

    if not proposals or not proposals.corrections:
        print("-> No corrections proposed.")
        return segments

    # Work on a copy so the original raw segments remain unchanged.
    refined_segments = [
        dict(seg)
        for seg in segments
    ]

    # Easy lookup from U-ID -> segment.
    segment_by_id = {
        seg.get("id"): seg
        for seg in refined_segments
    }

    print("Step 4: Running Guardrails & Verification...")

    for edit in proposals.corrections:
        uid = edit.utterance_id
        raw_span = edit.raw_span
        corr_span = edit.corrected_span

        # Check that the model referred to a real utterance.
        if uid not in segment_by_id:
            print(f"REJECTED: Unknown Utterance ID {uid}")
            continue

        segment = segment_by_id[uid]
        current_text = segment.get("text", "")

        # Only process if the exact raw span exists in the specified utterance.
        if raw_span not in current_text:
            print(
                f"REJECTED: Raw span not found in {uid}: "
                f"{raw_span}"
            )
            continue

        # 1. Deterministic Gate
        if not passes_deterministic_checks(raw_span, corr_span):
            continue

        # 2. Extract local context around the suspicious span.
        start_idx = current_text.find(raw_span)

        local_context = current_text[
            max(0, start_idx - 30):
            min(
                len(current_text),
                start_idx + len(raw_span) + 30
            )
        ]

        # 3. Contextual Gate
        if not verify_context(
            corr_span,
            local_context,
            topic_dict
        ):
            print(
                f"REJECTED: Failed contextual verification "
                f"({uid}: {corr_span})"
            )
            continue

        # 4. Apply passing edit only inside the identified utterance.
        print(
            f"ACCEPTED: {uid}: "
            f"{raw_span} -> {corr_span}"
        )

        segment["text"] = current_text.replace(
            raw_span,
            corr_span,
            1
        )

    return refined_segments