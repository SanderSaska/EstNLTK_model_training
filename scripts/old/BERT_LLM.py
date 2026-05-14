# Imports
import sys
import pathlib

# Add the project's root directory to the Python path
sys.path.append(pathlib.Path("../").resolve().as_posix())

# Configurations
seed = 42

# Paths
DATA_DIR = pathlib.Path("../data/")
ENC2017_ROOT = DATA_DIR / "enc2017"
UD_ET_EDT_ROOT = DATA_DIR / "ud_et_edt"
HOMONYMS_ROOT = DATA_DIR / "homonymous_word_forms"

ENC2017_DIRS = {
    "processed": ENC2017_ROOT / "processed",
    "raw": ENC2017_ROOT / "raw",
}

UD_ET_EDT_DIRS = {
    "processed": UD_ET_EDT_ROOT / "processed",
    "raw": UD_ET_EDT_ROOT / "raw",
}

HOMONYMS_DIRS = {
    "processed": HOMONYMS_ROOT / "processed",
    "annotations": HOMONYMS_ROOT / "annotations",
}

OUTPUT_DIR = pathlib.Path("../outputs/")

MODEL_DIR = pathlib.Path("../models/")

import os
import json
import random
import time
import pandas as pd
import estnltk
import tqdm

# from pydantic import BaseModel
import tiktoken
import tempfile
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai_cost_calculator import estimate_cost_typed
from typing import Any

load_dotenv(".env", verbose=True)
api_version = "2024-12-01-preview"
model_name = "gpt-4o"
deployment_name = "EstNLTK-gpt-4o"
endpoint = str(os.getenv("AZURE_ENDPOINT"))
api_key = str(os.getenv("OPENAI_API_KEY"))

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)
system_prompt = """You are an Estonian sentence rewriting assistant.
Your task is to select the best 5 candidates from the provided 100 BERT candidates to replace the marked word <...> in the sentence.

Rules:
- Use only the provided BERT candidates; do not invent new ones.
- Return candidates that fit naturally and grammatically in one of these cases: nominative, genitive, partitive, or additive/illative.
- Preserve the word's tense, number, case agreement, punctuation, and capitalisation.
- Prefer candidates whose form matches the marked word's grammatical role in the sentence.
- For proper names, keep only proper-name candidates that fit the same grammatical context.
- If fewer than 5 valid candidates exist, return as many as possible.
- Order candidates from best to worst fit.
- Do not repeat the original word unless it is the only valid option.
"""

# Structured output format for Azure OpenAI chat completions.
# Structured Outputs require a top-level object.
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "bert_llm_selection",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
            },
            "required": ["candidates"],
            "additionalProperties": False,
        },
    },
}

# Few-shot example to anchor the output format
few_shot_user = """Sentence: "Ehk oleks mõttekas ka mõni selleteemaline hoiatav <kampaania> korraldada," lisab punase autoga preili.
Candidates: ['aktsioon', 'üritus', 'kampaania', 'sündmus', 'saade', 'koosolek', 'kõne', 'päev', 'postitus', 'artikkel', 'video', 'näitus', 'asi', 'lugu', 'teade', 'lugemine', 'leht', 'liiklus', 'hoiatus', 'uudis', 'post', 'teavitus', 'loeng', 'komm', 'akt', 'õnnetus', 'sari', 'teema', 'tund', 'ettevõtmine', 'plakat', 'lõik', 'film', 'kirjutis', 'kirjandus', 'tekst', 'kontsert', 'seminar', 'märk', 'sõnum', 'konverents', 'üleskutse', 'foorum', 'festival', 'paraad', 'tegevus', 'sõit', 'reklaam', 'vms', 'eksperiment', 'paik', 'foto', 'kiri', 'nädal', 'ralli', 'juhtum', 'pilt', 'õppetund', 'ring', 'raamat', 'lehekülg', 'marss', 'jalutuskäik', 'pidu', 'koht', 'liik', 'küsitlus', 'mäng', 'traditsioon', 'kirjand', 'päeva', 'foor', 'avarii', 'õhtu', 'liikumine', 'kohtumine', 'arutelu', 'koolitus', 'võistlus', 'nali', 'liiklusõnnetus', 'ilm', 'tseremoonia', 'ajaleht', 'väljaanne', 'reis', 'näit', 'tee', 'materjal', 'test', 'püha', 'minut', 'kommentaarium', 'värk', 'näidis', 'katse', 'algatus', 'seeria', 'pühapäev', 'blogi']
"""

few_shot_assistant = json.dumps(
    {
        "candidates": [
            "aktsioon",
            "üritus",
            "sündmus",
            "näitus",
            "seminar",
        ],
    },
    ensure_ascii=False,
    indent=2,
)


# Create a user prompt template for the real sentences, including candidates list.
def create_user_prompt(sentence: str, word_span: tuple, candidates: list[str]) -> str:
    """Create a user prompt with marked sentence and candidate list."""
    # Mark the target word with angle brackets
    start, end = word_span
    marked_sentence = (
        sentence[:start] + "<" + sentence[start:end] + ">" + sentence[end:]
    )
    candidates_str = str(candidates)  # Simple list representation
    return f"""Sentence: {marked_sentence}
Candidates: {candidates_str}
"""


def build_messages(user_prompt: str) -> list[dict[str, str]]:
    """Build the chat-completions message list for one rewrite request."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": few_shot_user},
        {"role": "assistant", "content": few_shot_assistant},
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


def _atomic_write_json(file_path: pathlib.Path, data: list[dict[str, Any]]) -> None:
    """Write JSON atomically to avoid partial files on crash/interruption."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=file_path.parent,
            prefix=f"{file_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file_path = pathlib.Path(temp_file.name)

        # Atomic replace: after a crash you see either the old or the new file, never a partial one.
        os.replace(temp_file_path, file_path)
    finally:
        if temp_file_path is not None and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)


def _load_results_json(file_path: pathlib.Path) -> list[dict[str, Any]]:
    """Load results JSON safely and fail closed on corruption to prevent silent data loss."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return []

    with open(file_path, "r", encoding="utf-8") as file_handle:
        try:
            loaded = json.load(file_handle)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Failed to parse existing output file {file_path}. "
                "Refusing to overwrite it to avoid data loss."
            ) from error

    if not isinstance(loaded, list):
        raise RuntimeError(
            f"Expected a JSON list in {file_path}, got {type(loaded).__name__}. "
            "Refusing to overwrite it to avoid data loss."
        )
    return loaded


def append_result_to_json(
    file_path: pathlib.Path, record: dict[str, Any], id: str | None = None
) -> None:
    """Append or update one record in a JSON array using crash-safe atomic writes."""
    resolved_path = pathlib.Path(file_path)
    data = _load_results_json(resolved_path)

    if id is not None:
        # If ID is provided, update matching record; otherwise append.
        existing_record = next(
            (
                r
                for r in data
                if isinstance(r, dict)
                and r.get("id") is not None
                and str(r.get("id")) == str(id)
            ),
            None,
        )
        if existing_record:
            existing_record.update(record)
            # Remove stale error after a successful rewrite.
            existing_record.pop("error", None)
        else:
            data.append(record)
    else:
        data.append(record)

    _atomic_write_json(resolved_path, data)


def write_total_cost(file_path: pathlib.Path, total_cost: float) -> None:
    """Write the total cost to a separate txt file."""
    resolved_path = pathlib.Path(file_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        f.write(str(total_cost))


def rewrite_sentence_with_azure_openai(
    sentence_id: int,
    sentence: str,
    word_span: tuple,
    candidate_details: list[dict[str, Any]],
    metadata: dict[str, Any],
    candidates: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select best 5 BERT candidates via Azure OpenAI and derive predicted label from candidate scores."""
    user_prompt = create_user_prompt(
        sentence=sentence, word_span=word_span, candidates=candidates
    )
    messages = build_messages(user_prompt)

    # Request a structured JSON response from Azure OpenAI chat completions.
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        response_format=response_format,
        max_completion_tokens=512,
        top_p=0.95,
    )

    # Estimate cost for this response (best-effort).
    try:
        cost_result = estimate_cost_typed(response)
        cost_info = cost_result.as_dict()  # Convert to dict for JSON serialization
    except Exception as _err:
        # Don't fail the rewrite on estimator errors; keep a surfacable error message.
        cost_info = {"error": str(_err)}

    message = response.choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise RuntimeError(f"Model refused the request: {refusal}")

    # Parse the JSON response and enrich it with metadata for later analysis.
    parsed = parse_response_content(message.content or "{}")
    response_dict = response.to_dict()

    # Extract selected candidates and process BERT candidate details.
    selected_candidates = parsed.get("candidates", [])
    if not isinstance(selected_candidates, list):
        selected_candidates = []

    selected_candidate_details = [
        candidate
        for candidate in candidate_details
        if candidate.get("token") in selected_candidates
    ]

    # Normalise scores across selected candidates.
    selected_score_total = sum(
        candidate.get("score", 0.0) for candidate in selected_candidate_details
    )
    for candidate in selected_candidate_details:
        candidate_score = candidate.get("score", 0.0)
        candidate["normalised_score"] = (
            candidate_score / selected_score_total if selected_score_total > 0 else 0.0
        )

    # Derive morphological label predictions from selected candidate scores.
    predictions = {}
    for candidate in selected_candidate_details:
        resolved_form = candidate.get("resolved_form", [])
        if not resolved_form:
            continue
        label = resolved_form[0]
        predictions[label] = predictions.get(label, 0.0) + candidate["normalised_score"]

    predicted_label = max(predictions, key=predictions.get) if predictions else None

    # Enrich parsed response with metadata and analysis results.
    parsed["sentence_id"] = sentence_id
    parsed["candidate_details"] = selected_candidate_details
    parsed["predictions"] = predictions
    parsed["pred_label"] = predicted_label
    parsed["source_sentence"] = sentence
    parsed["target_word"] = metadata.get("target_word", "")

    if isinstance(word_span, list):
        parsed["word_span"] = word_span
    else:
        parsed["word_span"] = word_span.astype(
            int
        ).tolist()  # Convert numpy array to list for JSON serialization

    parsed["label"] = metadata.get("label", [])

    # Attach cost estimation metadata.
    response_dict["cost"] = cost_info

    return parsed, response_dict


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
    return dataset  # If no output file exists, return the entire dataset for processing


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


def rewrite_with_retry(
    sentence_id: int,
    sentence: str,
    word_span: tuple,
    candidate_details: list[dict[str, Any]],
    metadata: dict[str, Any],
    candidates: list[str],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call rewrite function with exponential backoff on transient errors."""
    for attempt in range(1, config["max_attempts"] + 1):
        try:
            return rewrite_sentence_with_azure_openai(
                sentence_id=sentence_id,
                sentence=sentence,
                word_span=word_span,
                candidate_details=candidate_details,
                metadata=metadata,
                candidates=candidates,
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


bert_mlm_candidates_filepath = (
    HOMONYMS_DIRS["processed"] / "homonyms_bert_mlm_candidate_details_100.jsonl"
)
with open(bert_mlm_candidates_filepath, "r", encoding="utf-8") as f:
    bert_mlm_candidates_data = [json.loads(line) for line in f]

bert_mlm_candidates_df = pd.DataFrame(bert_mlm_candidates_data)
display(bert_mlm_candidates_df.head(1))
output_file = OUTPUT_DIR / "gpt-4o-homonyms_LLM_MLM.json"  # Output file for results
# output_file_log = (
#     OUTPUT_DIR / "gpt-4o-homonyms_LLM_MLM_log.json"
# )  # More detailed log file for debugging and analysis
total_cost_txt = (
    OUTPUT_DIR / "gpt-4o-homonyms_LLM_MLM_total_cost.txt"
)  # File to store the total cost information
sample_df = bert_mlm_candidates_df.iloc[:].copy()  # Use N for testing
sample_df["index"] = sample_df.index.astype(
    int
)  # Ensure index is an integer for ID purposes

# gpt-4o has following rate limits:
# - 180 000 tokens per minute (TPM)
# - 1080 requests per minute (RPM)
config = {
    # Pacing configuration to avoid hitting rate limits
    "base_delay_seconds": round(60 / 1080, 2),  # RPM is 1080 for gpt-4o
    "jitter_seconds": 0.1,  # Add a small random jitter to avoid thundering herd issues
    # Retry configuration for transient failures
    "max_attempts": 6,
    "initial_backoff_seconds": 2.0,
    "max_backoff_seconds": 60.0,
}

# Start from the next unprocessed sentence based on the output file contents
unprocessed_df = get_unprocessed_dataset(sample_df, output_file)
if len(unprocessed_df) == 0:
    print("No unprocessed sentences found. All done!")
else:
    print(f"Total unprocessed sentences to process: {len(unprocessed_df)}")
    print("Next sentence to process:")
    display(unprocessed_df.head(1))

# Find processed records that are in the sample_df
if output_file.exists() and output_file.stat().st_size > 0:
    with open(output_file, "r", encoding="utf-8") as f:
        processed_data = json.load(f)
        processed_ids = [
            int(record["sentence_id"])
            for record in processed_data
            if "sentence_id" in record
        ]
        processed_sentences = sample_df[sample_df["index"].isin(processed_ids)]
        print(
            f"Number of sentences in sample_df that have already been processed: {len(processed_sentences)}"
        )
        print("Sample of processed sentences:")
        display(sample_df[sample_df["index"].isin(processed_ids)].head(10))
else:
    processed_sentences = (
        pd.DataFrame()
    )  # Default to empty DataFrame if no output file exists
    print("No output file found, so no sentences have been processed yet.")

# Now remove all the already processed sentences from the output file that are not in the sample_df
# if output_file.exists() and output_file.stat().st_size > 0:
#     with open(output_file, "r", encoding="utf-8") as f:
#         processed_data = json.load(f)
#         processed_ids = [
#             int(record["sentence_id"])
#             for record in processed_data
#             if "sentence_id" in record
#         ]
#         filtered_data = [
#             record
#             for record in processed_data
#             if int(record.get("sentence_id", -1)) in sample_df["index"].values
#         ]
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(filtered_data, f, ensure_ascii=False, indent=2)

# Find cumulative total cost from the output file
total_cost = 0.0
if total_cost_txt.exists() and total_cost_txt.stat().st_size > 0:
    with open(total_cost_txt, "r", encoding="utf-8") as f:
        total_cost = float(f.read().strip())
        print(f"Cumulative total cost from log file: ${total_cost:.6f}")
else:
    print("No log file found, so cumulative cost is $0.00.")
# Batch rewrite loop: call OpenAI and append each result to a JSON file
processed = len(processed_sentences)
max_cost = 20.0  # Set a maximum total cost threshold for the entire batch to avoid unexpected high costs during testing
with tqdm.tqdm(
    total=len(sample_df), desc="Processing sentences", initial=processed
) as pbar:
    for i, row in enumerate(
        zip(
            unprocessed_df["index"],
            unprocessed_df["sentence"],
            unprocessed_df["word_span"],
            unprocessed_df["word"],
            unprocessed_df["true_label"],
            unprocessed_df["candidate_details"],
        )
    ):
        sentence_id, sentence, word_span, word, label, candidate_details = row
        try:
            metadata = {
                "target_word": word,
                "label": label.tolist(),  # Convert numpy array to list for JSON serialization
            }
            candidates = [c.get("token") for c in candidate_details if "token" in c]
            result, response = rewrite_with_retry(
                sentence_id=sentence_id,
                sentence=sentence,
                word_span=word_span,
                candidate_details=candidate_details,
                metadata=metadata,
                candidates=candidates,
                config=config,
            )
            # Extract the parsed response (first element of tuple); ignore the response_dict for now
            parsed_result = result[0] if isinstance(result, tuple) else result
            append_result_to_json(output_file, parsed_result, id=sentence_id)
            # append_result_to_json(output_file_log, response, id=sentence_id) # Log the full response for debugging and analysis
            # Get the cost info from the result for logging
            cost_info = response.get("cost", {})
            if "error" in cost_info:
                tqdm.tqdm.write(
                    f"[{processed + 1}] Saved sentence_id={sentence_id} with cost estimation error: {cost_info['error']}"
                )
            else:
                total_cost_info = cost_info.get("total_cost", 0.0)
                total_cost += float(total_cost_info)
                write_total_cost(total_cost_txt, total_cost)
                # tqdm.tqdm.write(
                #     f"[{processed + 1}] Saved sentence_id={sentence_id} cost: {total_cost_info} total: {total_cost:.6f}"
                # )
        except Exception as exc:
            error_record = {
                "id": str(sentence_id),
                "source_sentence": sentence,
                "target_word": word,
                "word_span": word_span.astype(int).tolist(),
                "true_label": label.tolist(),
                "pred_label": None,
                "error": str(exc),
            }
            append_result_to_json(output_file, error_record, id=sentence_id)
            # append_result_to_json(
            #     output_file_log, {"error": str(exc)}, id=sentence_id
            # )  # Log the error details in the log file as well
            tqdm.tqdm.write(
                f"[{processed + 1}] Error on sentence_id={sentence_id}: {exc}"
            )
            print("Traceback:", exc.__traceback__)

        processed += 1
        # update postfix shown to the right of the bar
        pbar.set_postfix({"total_cost": f"${total_cost:.6f}", "processed": processed})
        pbar.update(1)

        if (
            total_cost > max_cost
        ):  # Stop the loop if total cost exceeds $20 to avoid unexpected high costs during testing
            print(
                f"Total cost exceeded ${max_cost}. Stopping the loop to avoid unexpected high costs during testing."
            )
            break

        # Additional pacing between successful/failed items to avoid bursty traffic.
        if i < len(unprocessed_df) - 1:  # Don't sleep after the last item
            sleep_s = config["base_delay_seconds"] + random.uniform(
                0.0, config["jitter_seconds"]
            )
            pbar.set_description_str(f"Sleeping {sleep_s:.2f}s")
            time.sleep(sleep_s)

print(f"Batch processing completed. Total cost: ${total_cost:.6f}")
print(f"Finished. Processed {processed} rows. Results appended to: {output_file}")
for index, row in tqdm.tqdm(llm_samples_df.iterrows(), total=llm_samples_df.shape[0]):
    id = row["id"]
    candidates = row.get("candidates", [])
    chosen = row.get("chosen", "")
    rewritten = row.get("rewritten", "")
    new_word_span = row.get("new_word_span", None)
    source_sentence = row["source_sentence"]
    target_word = row["target_word"]
    word_span = row["word_span"]
    label = row.get("label", [])[
        0
    ]  # Assuming label is a list and we take the first element as the form label
    # print(f"Source sentence: {source_sentence}")
    # print(f"Target word span: {word_span}")
    candidate_form_weight = (
        1 / len(candidates) if candidates else 0
    )  # Uniform weight for each candidate
    form_probabilities = {}
    for candidate in candidates:
        # Construct sentence with the candidate replacement
        candidate_sentence = (
            source_sentence[: word_span[0]]
            + candidate
            + source_sentence[word_span[1] :]
        )
        # print(f"Candidate sentence: {candidate_sentence}")
        # Create EstNLTK Text object and perform morphological analysis
        estnltk_text = estnltk.Text(candidate_sentence)
        estnltk_text.tag_layer("morph_analysis")
        # Extract the morphological tags for the candidate word
        morph_layer = estnltk_text.morph_analysis
        # Find the token in the morph layer that corresponds to the candidate replacement
        candidate_token = None
        for token in morph_layer:
            if (
                token.start <= word_span[0] < token.end
            ):  # Check if the token covers the start of the original word span
                candidate_token = token
                break
        if candidate_token:
            # Count up the form labels for the candidate
            number_of_labels = len(candidate_token.form) if candidate_token.form else 0
            if number_of_labels > 0:
                weight_per_label = candidate_form_weight / number_of_labels
                for candidate_label in candidate_token.form:
                    form_probabilities[candidate_label] = (
                        form_probabilities.get(candidate_label, 0) + weight_per_label
                    )
            else:
                form_probabilities["unknown"] = (
                    form_probabilities.get("unknown", 0) + candidate_form_weight
                )

    best_form_label = (
        max(form_probabilities, key=form_probabilities.get)
        if form_probabilities
        else "unknown"
    )
    # Create a record for the chosen candidate and its features
    candidate_record = {
        "id": id,
        "candidates": candidates,
        "pred_label": best_form_label,
        "true_label": label,
        "predicted_form_distribution": form_probabilities,
        "source_sentence": source_sentence,
        "target_word": target_word,
        "word_span": word_span,
    }
    new_records.append(candidate_record)

# Create a new DataFrame from the predicted form labels and their distributions and save to JSON
homonym_results_df = pd.DataFrame(new_records)
homonym_results_df_path = (
    HOMONYMS_DIRS["processed"] / f"homonyms_llm_mlm_predictions_v2_{max_rows}.parquet"
)
homonym_results_df.to_parquet(homonym_results_df_path, index=False)
