# Meeting Assistant

An end-to-end AI meeting assistant that converts an English meeting
recording into a refined transcript and an evidence-grounded meeting
record containing meeting information such as summaries, decisions, and
action items.

The pipeline separates speech recognition, transcript refinement,
meeting-information extraction, evidence verification, and final
rendering so that each stage has a clear responsibility.

------------------------------------------------------------------------

## Overview

``` text
Audio
  ↓
16 kHz Mono WAV
  ↓
NVIDIA Whisper large-v3
  ↓
Raw Transcript
+ Timestamps
+ Utterance IDs
  ↓
Global Topic Pass
  ↓
Local Topic Pass
  ↓
Localized ASR Correction
  ↓
Deterministic Guardrails
  ↓
Contextual Verification
  ↓
Refined Transcript
  ↓
Meeting Record Extraction
  ↓
Evidence Quality Gate
  ↓
Verified Record
  ├── JSON
  └── Markdown
```

The raw transcript is preserved separately from the refined transcript.
Stable utterance IDs (`U001`, `U002`, ...) provide the link between
transcript segments and evidence used by downstream meeting-record
extraction.

------------------------------------------------------------------------

## Key Features

-   English meeting audio processing
-   Audio conversion to a canonical 16 kHz mono WAV format
-   NVIDIA hosted Whisper large-v3 speech-to-text
-   Timestamped transcript segments
-   Stable utterance IDs for evidence grounding
-   Global meeting-topic analysis
-   Local discussion-topic analysis
-   Localized ASR error correction
-   Exact raw-span and utterance-ID validation
-   Deterministic preservation guardrails
-   Contextual LLM verification of proposed corrections
-   Refined transcript with preserved segment identity and timestamps
-   Evidence-grounded meeting-information extraction
-   Evidence quality filtering
-   Machine-readable JSON output
-   Human-readable Markdown meeting minutes

------------------------------------------------------------------------

## Pipeline

### 1. Audio Processing

The input audio is validated and converted into a canonical 16 kHz mono
WAV representation.

This provides a consistent input format for the speech-to-text stage.

### 2. Speech-to-Text

The system uses NVIDIA's hosted Whisper large-v3 service through the
NVIDIA Riva/gRPC client.

The STT stage performs English recognition and requests:

-   Automatic punctuation
-   One best recognition result
-   Word-time offsets

The returned word offsets are used to derive the start and end
timestamps of each transcript segment.

### 3. Raw Transcript and Utterance IDs

Each recognized segment receives a stable sequential identifier:

``` text
[U001] ...
[U002] ...
[U003] ...
```

A raw transcript artifact is then stored containing:

-   Audio duration
-   Segment IDs
-   Segment timestamps
-   Raw ASR text
-   Full raw transcript

The raw segment representation remains separate from the refined
representation.

### 4. Global Topic Analysis

The complete raw transcript is passed through a global topic analysis
step.

It identifies:

-   The primary meeting topic
-   Relevant technical terms
-   Acronyms
-   Domain vocabulary

This provides meeting-level context for the refinement stage.

### 5. Local Topic Analysis

The transcript is divided into contiguous discussion regions based on
changes in the specific subject being discussed.

Each local region contains:

-   Starting utterance ID
-   Ending utterance ID
-   Local discussion topic
-   Relevant keywords

Every transcript utterance is mapped to its local topic, with the global
topic available as fallback context.

### 6. Localized ASR Correction

The correction model receives the transcript together with its stable
U-IDs and the available global/local topic context.

It is instructed to identify specific ASR errors and return:

-   The utterance ID containing the error
-   The exact raw span
-   The proposed corrected span

The correction stage is localized: it does not rewrite the complete
transcript.

### 7. Deterministic Guardrails

Before a proposed correction can be applied, deterministic preservation
checks are performed.

The current checks protect:

-   Numbers
-   Dates
-   Times
-   Versions
-   Percentages
-   Negation
-   Uncertainty markers
-   Commitment-level markers

A correction is rejected if it changes protected information.

### 8. Contextual Verification

Corrections that pass the deterministic checks are sent through a
contextual LLM verification step.

The verifier receives:

-   The proposed correction
-   Local text surrounding the original span
-   The meeting topic

The correction is accepted only when it is supported by the surrounding
context.

Accepted corrections are applied only to the copied refined segment.

### 9. Refined Transcript

The refined transcript preserves:

-   Original utterance IDs
-   Segment timestamps
-   Segment boundaries
-   Approved text corrections

The pipeline writes both human-readable and structured refined
transcript artifacts.

### 10. Meeting-Record Extraction

The refined transcript is formatted with U-IDs and passed to the
meeting-documentation stage.

The extraction stage produces structured meeting information such as:

-   Meeting summary
-   Decisions
-   Action items

Extracted items are associated with transcript evidence so their claims
can be checked against the source transcript.

### 11. Evidence Quality Gate

The extracted record passes through a verification and filtering stage.

The quality gate checks whether generated meeting items are supported by
their cited transcript evidence and filters unsupported items before the
final record is produced.

### 12. Final Rendering

The verified meeting record is serialized as JSON.

The same verified record is used to generate the Markdown meeting
minutes.

This keeps the structured record and human-readable minutes consistent.

------------------------------------------------------------------------

## Evidence Grounding

The pipeline uses stable utterance IDs as its evidence reference layer.

For example:

``` text
[U012] We discussed the deployment schedule.
[U013] The model should be deployed next week.
[U014] The engineering team will prepare the deployment package.
```

A downstream meeting item can reference the relevant utterances:

``` text
Source IDs: U013, U014
```

This allows extracted decisions and action items to be traced back to
specific transcript segments.

------------------------------------------------------------------------

## Output Files

For an input such as:

``` text
data/sample/meeting.mp3
```

the pipeline creates an output directory based on the audio filename.

The principal artifacts are:

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `input_16k.wav`                     Canonical 16 kHz mono audio used by
                                      STT

  `raw_transcript.json`               Raw STT transcript with timestamps,
                                      U-IDs, and full text

  `refined_transcript.txt`            Combined refined transcript

  `refined_transcript_with_ids.txt`   Refined transcript with U-IDs

  `refined_transcript.json`           Structured refined transcript with
                                      timestamps and full text

  `record.json`                       Verified structured meeting record

  `record.md`                         Human-readable meeting minutes
  -----------------------------------------------------------------------

Example:

``` text
outputs/
└── meeting/
    ├── input_16k.wav
    ├── raw_transcript.json
    ├── refined_transcript.txt
    ├── refined_transcript_with_ids.txt
    ├── refined_transcript.json
    ├── record.json
    └── record.md
```

------------------------------------------------------------------------

## Project Structure

``` text
meeting-assistant/
│
├── data/
├── docs/
├── outputs/
├── prompts/
│
├── scripts/
│   └── pipeline.py
│
├── src/
│   ├── audio.py
│   ├── config.py
│   ├── llm_client.py
│   ├── metrics.py
│   ├── schemas.py
│   │
│   ├── stt/
│   │   └── nvidia_backend.py
│   │
│   ├── refine/
│   │   └── refiner.py
│   │
│   └── document/
│       └── extractor.py
│
├── .env.example
├── requirements.txt
└── README.md
```

### Main Components

  -----------------------------------------------------------------------
  Component                           Responsibility
  ----------------------------------- -----------------------------------
  `scripts/pipeline.py`               End-to-end pipeline orchestration

  `src/audio.py`                      Audio conversion and preprocessing

  `src/stt/`                          Speech-to-text backend

  `src/stt/nvidia_backend.py`         NVIDIA Riva/Whisper STT
                                      implementation

  `src/refine/refiner.py`             Topic analysis, correction
                                      proposals, guardrails,
                                      verification, and refinement

  `src/schemas.py`                    Structured data models

  `src/document/extractor.py`         Meeting-record extraction, evidence
                                      verification, and Markdown
                                      rendering

  `src/config.py`                     Model/service configuration

  `src/llm_client.py`                 LLM invocation interface
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Requirements

The project dependencies are listed in:

``` text
requirements.txt
```

The pipeline also requires access to the configured NVIDIA
speech-to-text service and the configured LLM service.

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone <repository-url>
cd meeting-assistant
```

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it.

### Windows PowerShell

``` powershell
.venv\Scripts\Activate.ps1
```

### Linux / macOS

``` bash
source .venv/bin/activate
```

Install the project dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Environment Configuration

Create the environment configuration from the provided example:

``` text
.env.example
```

The STT configuration includes the NVIDIA service credentials and
connection settings used by the NVIDIA backend.

The refinement stage uses the configured refinement model.

Do not commit API keys or other credentials to the repository.

------------------------------------------------------------------------

## Running the Pipeline

The main entry point is:

``` text
scripts/pipeline.py
```

Run the complete pipeline with:

``` bash
python scripts/pipeline.py --audio path/to/meeting.mp3
```

An output directory can also be specified:

``` bash
python scripts/pipeline.py --audio path/to/meeting.mp3 --out outputs
```

The pipeline performs the complete workflow in one run:

``` text
Audio
→ preprocessing
→ STT
→ raw transcript
→ refinement
→ meeting-record extraction
→ evidence verification
→ JSON + Markdown
```

------------------------------------------------------------------------

## Error Handling

The pipeline handles failures at the stage where they occur.

Examples include:

-   Missing audio files
-   Audio preprocessing failures
-   STT failures
-   Invalid utterance IDs proposed during refinement
-   Proposed raw spans that do not exist in the referenced utterance
-   Corrections rejected by deterministic preservation checks
-   Corrections rejected by contextual verification
-   Invalid meeting-record extraction results
-   Unsupported meeting items filtered by the evidence quality gate

------------------------------------------------------------------------

## Design Principles

### Preserve the source transcript

The raw ASR transcript is kept separately from the refined transcript.

### Correct locally

The refinement stage changes specific identified spans instead of
rewriting the entire transcript.

### Use context before changing ASR output

Global and local discussion topics provide context for deciding whether
a proposed correction is appropriate.

### Protect important information deterministically

Numbers, dates, times, versions, percentages, negation, uncertainty, and
commitment-level information are checked by deterministic rules.

### Ground generated claims

Meeting-record items are associated with transcript evidence so that
generated claims can be checked against the meeting transcript.

### One verified record, multiple views

The verified structured meeting record is the source used for both JSON
and Markdown outputs.

------------------------------------------------------------------------

## Evaluation

The architecture provides clear evaluation boundaries:

1.  **Raw STT** --- quality of the initial transcript.
2.  **Transcript refinement** --- quality of localized ASR corrections.
3.  **Meeting-record extraction** --- quality of summaries, decisions,
    and action items.
4.  **Evidence support** --- whether generated items are grounded in
    transcript evidence.
5.  **End-to-end pipeline** --- correctness of the complete output from
    audio to final meeting record.

The separate artifacts make it possible to inspect each stage
independently.

------------------------------------------------------------------------

## Documentation

For the detailed technical description of the architecture, data flow,
refinement process, evidence grounding, and implementation, see:

``` text
docs/Meeting_Assistant_Current_Technical_Document.pdf
```

The README is intended as the quick-start and repository guide; the
technical document contains the deeper implementation reference.

------------------------------------------------------------------------

## Team

This project is organized into three major responsibilities:

-   **Speech-to-Text / Audio Pipeline**
-   **Transcript Refinement**
-   **Meeting Intelligence / Documentation**
