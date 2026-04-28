import json
import random
import time
from typing import Any
from openai_cost_calculator import estimate_cost_typed


# System prompt: enforce Estonian rewriting behaviour, case conditioning, and JSON output
system_prompt = """You are an Estonian sentence rewriting assistant.

Replace exactly one marked token in angle brackets <...> with a context-appropriate alternative.

Rules:
- Preserve tense, number, case, agreement, capitalisation, punctuation, and word order as much as possible.
- Infer the token’s grammatical role and morphology from the sentence context.
- Generate 10 alternatives that fit the context and use only these cases: nominative, genitive, partitive, or additive/illative (both fit the same role in this task).
- If the token is a proper name, replace it with another plausible proper name that still fits the required case.
- Do not repeat the original token unless it is the only valid option.
"""

# Structured output format for Azure OpenAI chat completions.
# Structured Outputs require a top-level object, so the list is wrapped in a candidates field.
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "candidates",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 10,
                    "maxItems": 10,
                }
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
    },
}

# One-shot example to anchor the output format
one_shot_user = """Sentence: Ma näen <maja>."""

one_shot_assistant = json.dumps(
    {
        "candidates": [
            "ehitist",
            "hoonet",
            "elamut",
            "rajatist",
            "korterit",
            "kodu",
            "eluaset",
            "hoonestust",
            "majapidamist",
            "häärberit",
        ],
    },
    ensure_ascii=False,
    indent=2,
)

# Example real input sentence
# user_prompt = """Sentence: Samas on kõik uue kodu lähistel asuvad koolid sellised, mis võtavad <gümnaasiumi> vastu katsetega."""


# Create a user prompt template for the real sentences, using the same format as the few-shot example
def create_user_prompt(sentence, word_span):
    # Mark the target word with angle brackets
    start, end = word_span
    marked_sentence = (
        sentence[:start] + "<" + sentence[start:end] + ">" + sentence[end:]
    )
    return f"""Sentence: {marked_sentence}"""


def build_messages(user_prompt: str) -> list[dict[str, str]]:
    """Build the chat-completions message list for one rewrite request."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": one_shot_user},
        {"role": "assistant", "content": one_shot_assistant},
        {"role": "user", "content": user_prompt},
    ]


def parse_response_content(content: str) -> dict[str, Any]:
    """Parse and validate the JSON payload returned by Azure OpenAI."""
    parsed = json.loads(content or "{}")
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from the model response.")
    if "candidates" not in parsed or not isinstance(parsed["candidates"], list):
        raise ValueError("Model response did not contain a candidates list.")
    return parsed


def append_result_to_json(file_path, record, id=None, search_id="sentence_id"):
    """Append one record to a JSON array stored in file_path."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure any structured cost object is converted into plain JSON-serialisable data.
    if isinstance(record, dict) and "_cost" in record:
        cost_value = record["_cost"]
        if hasattr(cost_value, "as_dict"):
            record = dict(record)
            record["_cost"] = cost_value.as_dict()
    # If the file exists and is non-empty, load the existing data; otherwise start with an empty list
    if file_path.exists() and file_path.stat().st_size > 0:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    if not isinstance(data, list):
        data = []
    if id is not None and search_id is not None:
        # If ID is provided, check if a record with the same ID already exists and update it; otherwise append the new record
        existing_record = next(
            (r for r in data if int(r.get(search_id)) == int(id)), None
        )
        if existing_record:
            existing_record.update(record)
            # Remove error field if it exists, since we have a successful rewrite now
            existing_record.pop("error", None)
        else:
            data.append(
                record
            )  # Append the new record if no existing record with the same ID is found
    else:
        # If no ID is provided, put the record at the end of the list
        data.append(record)
    # Write the updated list back to the file with pretty formatting
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rewrite_sentence_with_azure_openai(sentence_id, sentence, word_span, metadata):
    """Rewrite one sentence by replacing the marked target word via Azure OpenAI."""
    user_prompt = create_user_prompt(sentence=sentence, word_span=word_span)
    messages = build_messages(user_prompt)

    # Request a structured JSON response from Azure OpenAI chat completions.
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        response_format=response_format,
        max_completion_tokens=1000,
        top_p=0.95,
    )
    # Estimate cost for this response (best-effort). The estimator expects the full response object.
    try:
        cost_info = estimate_cost_typed(response)
    except Exception as _err:
        # Don't fail the rewrite on estimator errors; keep a surfacable error message.
        cost_info = {"error": str(_err)}
    message = response.choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise RuntimeError(f"Model refused the request: {refusal}")

    # Parse the JSON response and enrich it with metadata for later analysis.
    parsed = parse_response_content(message.content or "{}")

    # Keep useful metadata for later analysis/debugging.
    parsed["sentence_id"] = str(sentence_id)
    parsed["source_sentence"] = sentence
    parsed["target_word"] = metadata.get("target_word", "")
    if isinstance(word_span, list):
        parsed["word_span"] = word_span
    else:
        parsed["word_span"] = word_span.astype(
            int
        ).tolist()  # Convert numpy array to list for JSON serialization
    parsed["label"] = metadata.get("label", [])
    # Attach cost estimation metadata (not part of the structured schema)
    parsed["_cost"] = cost_info

    return parsed, response


# Backwards-compatible alias for any existing callers.
rewrite_sentence_with_genai = rewrite_sentence_with_azure_openai


def get_latest_processed_id(output_file):
    """Get the highest sentence ID that has already been processed and saved in the output file."""
    if output_file.exists() and output_file.stat().st_size > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return max(int(record["id"]) for record in data if "id" in record)
    return -1  # No valid records found, start from the beginning


def get_unprocessed_dataset(dataset, output_file):
    """Filter the dataset to include only records that have errors or have not been processed yet, based on the output file."""
    if output_file.exists() and output_file.stat().st_size > 0:
        unprocessed_records = []
        processed_records = []
        # First, get the list of sentences that have errors in the output file
        with open(output_file, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
            for record in processed_data:
                if "error" in record:
                    unprocessed_records.append(record["source_sentence"])
                else:
                    processed_records.append(record["source_sentence"])
        # Now filter the original dataset to include only records that are either unprocessed or have errors
        filtered_dataset = dataset[
            dataset["sentence"].isin(unprocessed_records)
            | ~dataset["sentence"].isin(processed_records)
        ]
        return filtered_dataset
    else:
        return dataset  # If no output file exists, return the entire dataset as unprocessed


def is_transient_error(exc: Exception) -> bool:
    """Heuristically detect retryable errors such as rate limit and network hiccups."""
    message = str(exc).lower()
    transient_markers = [
        "429",
        "rate",
        "quota",
        "too many requests",
        "timeout",
        "timed out",
        "connection",
        "temporar",
        "unavailable",
        "503",
        "502",
        "504",
        "internal",
    ]
    return any(marker in message for marker in transient_markers)


def rewrite_with_retry(sentence_id, sentence, word_span, metadata, config):
    """Call rewrite function with exponential backoff on transient errors."""
    for attempt in range(1, config["max_attempts"] + 1):
        try:
            return rewrite_sentence_with_azure_openai(
                sentence_id=sentence_id,
                sentence=sentence,
                word_span=word_span,
                metadata=metadata,
            )
        except Exception as exc:
            # should_retry = is_transient_error(exc) and attempt < config["max_attempts"]
            should_retry = (
                attempt < config["max_attempts"]
            )  # Retry on all exceptions for robustness, but still limit the number of attempts
            if not should_retry:
                raise

            backoff = min(
                config["max_backoff_seconds"],
                config["initial_backoff_seconds"] * (2 ** (attempt - 1)),
            )
            sleep_s = backoff + random.uniform(0.0, config["jitter_seconds"])
            print(
                f"Retry {attempt}/{config['max_attempts'] - 1} for sentence_id={sentence_id} "
                f"after transient error: {exc}. Sleeping {sleep_s:.2f}s..."
            )
            time.sleep(sleep_s)
