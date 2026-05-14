import json
import random
import time
from pathlib import Path
from typing import Any


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

BATCH_CHAT_COMPLETIONS_URL = "/v1/chat/completions"
BATCH_COMPLETION_WINDOW = "24h"
BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

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


def serialise_word_span(word_span: Any) -> list[int]:
    """Convert a span-like object into a JSON-serialisable list of integers."""
    if isinstance(word_span, list):
        return [int(value) for value in word_span]
    if hasattr(word_span, "astype"):
        return word_span.astype(int).tolist()
    return [int(value) for value in word_span]


def build_rewrite_request_body(
    sentence: str,
    word_span: Any,
    model_name: str,
) -> dict[str, Any]:
    """Build the chat-completions payload for a single rewrite request."""
    user_prompt = create_user_prompt(sentence=sentence, word_span=word_span)
    return {
        "model": model_name,
        "messages": build_messages(user_prompt),
        "response_format": response_format,
        "max_completion_tokens": 512,
        "top_p": 0.95,
    }


def build_rewrite_batch_request_line(
    sentence_id,
    sentence: str,
    word_span: Any,
    metadata: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Build one JSONL line for the Batch API."""
    return {
        "custom_id": str(sentence_id),
        "method": "POST",
        "url": BATCH_CHAT_COMPLETIONS_URL,
        "body": build_rewrite_request_body(
            sentence=sentence,
            word_span=word_span,
            model_name=model_name,
        ),
    }


def build_rewrite_manifest_record(
    sentence_id,
    sentence: str,
    word_span: Any,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build a local manifest record used to merge batch outputs back to the source rows."""
    return {
        "sentence_id": str(sentence_id),
        "source_sentence": sentence,
        "target_word": metadata.get("target_word", ""),
        "word_span": serialise_word_span(word_span),
        "label": metadata.get("label", []),
    }


def build_rewrite_tasks_from_dataframe(
    dataframe,
    *,
    sentence_id_column: str = "index",
    sentence_column: str = "sentence",
    word_span_column: str = "word_span",
    target_word_column: str = "word",
    label_column: str = "label",
) -> list[dict[str, Any]]:
    """Convert a dataframe-style object into batch rewrite tasks.

    The helper expects the same columns used by the notebook pipeline and keeps
    the batch-building code independent from pandas imports in this module.
    """
    tasks = []

    for _, row in dataframe.iterrows():
        label_value = row[label_column]
        if hasattr(label_value, "tolist"):
            label_value = label_value.tolist()

        tasks.append(
            {
                "sentence_id": row[sentence_id_column],
                "sentence": row[sentence_column],
                "word_span": row[word_span_column],
                "metadata": {
                    "target_word": row[target_word_column],
                    "label": label_value,
                },
            }
        )

    return tasks


def write_jsonl(file_path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a newline-delimited JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def build_rewrite_batch_files(
    tasks: list[dict[str, Any]],
    batch_input_path: Path,
    manifest_path: Path,
    model_name: str,
) -> None:
    """Create the Batch API input file and a local manifest for later result merging."""
    batch_lines = []
    manifest_records = []

    for task in tasks:
        sentence_id = task["sentence_id"]
        sentence = task["sentence"]
        word_span = task["word_span"]
        metadata = task.get("metadata", {})

        batch_lines.append(
            build_rewrite_batch_request_line(
                sentence_id=sentence_id,
                sentence=sentence,
                word_span=word_span,
                metadata=metadata,
                model_name=model_name,
            )
        )
        manifest_records.append(
            build_rewrite_manifest_record(
                sentence_id=sentence_id,
                sentence=sentence,
                word_span=word_span,
                metadata=metadata,
            )
        )

    write_jsonl(batch_input_path, batch_lines)
    batch_input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_records, handle, ensure_ascii=False, indent=2)


def submit_rewrite_batch(
    client,
    batch_input_path: Path,
    *,
    completion_window: str = BATCH_COMPLETION_WINDOW,
    endpoint: str = BATCH_CHAT_COMPLETIONS_URL,
    metadata: dict[str, Any] | None = None,
):
    """Upload a JSONL input file and create an OpenAI batch job."""
    with open(batch_input_path, "rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")

    return client.batches.create(
        input_file_id=input_file.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {},
    )


def wait_for_batch_completion(
    client,
    batch_id: str,
    *,
    poll_interval_seconds: int = 60,
    timeout_seconds: int = 24 * 60 * 60,
):
    """Poll a batch until it reaches a terminal state."""
    started_at = time.monotonic()

    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)

        if status in BATCH_TERMINAL_STATUSES:
            if status != "completed":
                raise RuntimeError(f"Batch {batch_id} finished with status={status}")
            return batch

        elapsed_seconds = time.monotonic() - started_at
        if elapsed_seconds > timeout_seconds:
            raise TimeoutError(
                f"Batch {batch_id} did not finish within {timeout_seconds} seconds"
            )

        print(
            f"Batch {batch_id} status={status}; sleeping {poll_interval_seconds}s before polling again..."
        )
        time.sleep(poll_interval_seconds)


def download_batch_file(client, file_id: str, destination_path: Path) -> Path:
    """Download a Batch output/error file to disk."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    file_response = client.files.content(file_id)
    destination_path.write_text(file_response.text(), encoding="utf-8")
    return destination_path


def load_manifest_lookup(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Load the local manifest and index it by sentence_id."""
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest_records = json.load(handle)

    return {
        str(record["sentence_id"]): record
        for record in manifest_records
        if isinstance(record, dict) and "sentence_id" in record
    }


def parse_batch_response_line(
    line: str,
    manifest_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Parse one Batch API response line and merge it with the local manifest."""
    payload = json.loads(line)
    custom_id = str(payload.get("custom_id", ""))
    manifest = manifest_lookup.get(custom_id, {})
    response = payload.get("response") or {}
    error = payload.get("error") or None

    if error:
        return {
            **manifest,
            "sentence_id": custom_id,
            "batch_custom_id": custom_id,
            "batch_id": payload.get("id", ""),
            "error": error.get("message", json.dumps(error, ensure_ascii=False)),
        }

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return {
            **manifest,
            "sentence_id": custom_id,
            "batch_custom_id": custom_id,
            "batch_id": payload.get("id", ""),
            "error": "Batch response did not contain any choices.",
        }

    message = choices[0].get("message") or {}
    refusal = message.get("refusal")
    if refusal:
        return {
            **manifest,
            "sentence_id": custom_id,
            "batch_custom_id": custom_id,
            "batch_id": payload.get("id", ""),
            "error": f"Model refused the request: {refusal}",
        }

    parsed = parse_response_content(message.get("content") or "{}")
    parsed.update(manifest)
    parsed["sentence_id"] = custom_id
    parsed["batch_custom_id"] = custom_id
    parsed["batch_id"] = payload.get("id", "")
    parsed["request_id"] = response.get("request_id", "")
    parsed["status_code"] = response.get("status_code", 0)
    parsed["usage"] = body.get("usage", {})
    return parsed


def ingest_batch_result_files(
    batch_output_path: Path,
    manifest_path: Path,
    output_file: Path,
    output_file_log: Path | None = None,
) -> None:
    """Merge Batch API output/error files back into the JSON result files used by the notebook."""
    manifest_lookup = load_manifest_lookup(manifest_path)

    for source_path, search_id in ((batch_output_path, "custom_id"),):
        if not source_path.exists() or source_path.stat().st_size == 0:
            continue

        with open(source_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                raw_record = json.loads(line)
                merged_record = parse_batch_response_line(line, manifest_lookup)
                append_result_to_json(
                    output_file,
                    merged_record,
                    id=merged_record.get("sentence_id"),
                    search_id="sentence_id",
                )

                if output_file_log is not None:
                    append_result_to_json(
                        output_file_log,
                        raw_record,
                        id=raw_record.get("custom_id"),
                        search_id=search_id,
                    )


def run_rewrite_batch(
    client,
    tasks: list[dict[str, Any]],
    *,
    model_name: str,
    batch_input_path: Path,
    manifest_path: Path,
    output_file: Path,
    output_file_log: Path | None = None,
    batch_metadata: dict[str, Any] | None = None,
    completion_window: str = BATCH_COMPLETION_WINDOW,
    poll_interval_seconds: int = 60,
    timeout_seconds: int = 24 * 60 * 60,
) -> Any:
    """Submit a rewrite batch, wait for completion, and ingest the output files."""
    build_rewrite_batch_files(
        tasks=tasks,
        batch_input_path=batch_input_path,
        manifest_path=manifest_path,
        model_name=model_name,
    )

    batch = submit_rewrite_batch(
        client,
        batch_input_path,
        completion_window=completion_window,
        metadata=batch_metadata,
    )
    completed_batch = wait_for_batch_completion(
        client,
        batch.id,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
    )

    batch_dir = batch_input_path.parent
    if getattr(completed_batch, "output_file_id", None):
        batch_output_path = batch_dir / f"{batch_input_path.stem}_output.jsonl"
        download_batch_file(
            client,
            completed_batch.output_file_id,
            batch_output_path,
        )
        ingest_batch_result_files(
            batch_output_path=batch_output_path,
            manifest_path=manifest_path,
            output_file=output_file,
            output_file_log=output_file_log,
        )

    if getattr(completed_batch, "error_file_id", None):
        batch_error_path = batch_dir / f"{batch_input_path.stem}_error.jsonl"
        download_batch_file(
            client,
            completed_batch.error_file_id,
            batch_error_path,
        )
        ingest_batch_result_files(
            batch_output_path=batch_error_path,
            manifest_path=manifest_path,
            output_file=output_file,
            output_file_log=output_file_log,
        )

    return completed_batch


def append_result_to_json(file_path, record, id=None, search_id="sentence_id"):
    """Append one record to a JSON array stored in file_path."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
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
            (
                r
                for r in data
                if isinstance(r, dict)
                and r.get(search_id) is not None
                and str(r.get(search_id)) == str(id)
            ),
            None,
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


def rewrite_sentence_with_azure_openai(
    sentence_id,
    sentence,
    word_span,
    metadata,
    model_name,
):
    """Build the Batch API request line for one sentence rewrite."""
    request_line = build_rewrite_batch_request_line(
        sentence_id=sentence_id,
        sentence=sentence,
        word_span=word_span,
        metadata=metadata,
        model_name=model_name,
    )
    manifest_record = build_rewrite_manifest_record(
        sentence_id=sentence_id,
        sentence=sentence,
        word_span=word_span,
        metadata=metadata,
    )
    return request_line, manifest_record


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


def rewrite_with_retry(sentence_id, sentence, word_span, metadata, config, model_name):
    """Build one batch task with exponential backoff on transient errors."""
    for attempt in range(1, config["max_attempts"] + 1):
        try:
            return rewrite_sentence_with_azure_openai(
                sentence_id=sentence_id,
                sentence=sentence,
                word_span=word_span,
                metadata=metadata,
                model_name=model_name,
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
