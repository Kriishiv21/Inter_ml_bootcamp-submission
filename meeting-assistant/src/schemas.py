from pydantic import BaseModel, Field
from typing import List, Optional

# --- STAGE 2: REFINEMENT SCHEMAS ---

class TopicDictionary(BaseModel):
    global_topic: str = Field(description="The broad topic of the meeting region")
    keywords: List[str] = Field(description="Domain-specific terms, acronyms, or entities found")

class LocalTopicSegment(BaseModel):
    start_utterance_id: str = Field(description="First U-ID in this local topic region")
    end_utterance_id: str = Field(description="Last U-ID in this local topic region")
    local_topic: str = Field(description="Short description of the local discussion topic")
    keywords: List[str] = Field(default_factory=list, description="Technical terms relevant to this local topic")

class LocalTopicMap(BaseModel):
    segments: List[LocalTopicSegment]

class ProposedCorrection(BaseModel):
    utterance_id: str = Field(description="The U-ID of the utterance containing the suspicious span, e.g. U012")
    raw_span: str = Field(description="The exact text from that utterance that is suspicious")
    corrected_span: str = Field(description="The proposed corrected technical term or phrase")
    reason: str = Field(description="Why this is an ASR error")

class CorrectionList(BaseModel):
    corrections: List[ProposedCorrection]

class ContextualVerification(BaseModel):
    is_supported: bool = Field(description="True if the corrected_span logically fits the global and local topic")
    reason: str

# --- STAGE 3: MOM & EXTRACTION SCHEMAS ---

class ActionItem(BaseModel):
    description: str = Field(description="The specific task to be done")
    owner: str = Field(description="The person assigned. Use 'Unassigned' if not explicitly stated.")
    deadline: str = Field(description="The due date. Use 'Unspecified' if not stated.")
    evidence_utterance_ids: List[str] = Field(description="List of U-IDs (e.g., ['U012', 'U013']) supporting this task")

class KeyDecision(BaseModel):
    decision: str = Field(description="An agreed-upon decision. Do not include unconfirmed proposals.")
    evidence_utterance_ids: List[str] = Field(description="List of U-IDs supporting this decision")
class EvidenceVerification(BaseModel):
    is_supported: bool = Field(description="Whether the cited utterances support the claim")
    reason: str = Field(description="Why the evidence does or does not support the claim")
    unsupported_utterance_ids: List[str] = Field(description="Cited U-IDs that do not support the claim")
class MeetingRecord(BaseModel):
    summary: str = Field(description="A concise summary of the discussion")
    decisions: List[KeyDecision]
    action_items: List[ActionItem]