import json
from transformers import AutoConfig


def compare_label_lists(labels_json_path: str, model_dir: str):
    with open(labels_json_path, "r", encoding="utf-8") as f:
        user_labels = list(json.load(f))  # list of label strings

    cfg = AutoConfig.from_pretrained(model_dir)
    cfg_id2label = getattr(cfg, "id2label", None)
    cfg_label2id = getattr(cfg, "label2id", None)

    # Build model label list in index order if possible
    model_labels = []
    model_label2id = {}
    model_id2label = {}
    if cfg_id2label:
        model_id2label = {int(k): str(v) for k, v in cfg_id2label.items()}
        model_labels = [model_id2label[i] for i in sorted(model_id2label.keys())]
        model_label2id = {v: k for k, v in model_id2label.items()}
    elif cfg_label2id:
        model_label2id = {str(k): int(v) for k, v in cfg_label2id.items()}
        # invert to ordered list if contiguous
        max_id = max(model_label2id.values())
        model_labels = [None] * (max_id + 1)
        for lab, idx in model_label2id.items():
            model_labels[idx] = lab
        model_id2label = {
            i: lab for i, lab in enumerate(model_labels) if lab is not None
        }
    else:
        print("Model config has no id2label/label2id mapping.")
        model_labels = []
        model_label2id = {}
        model_id2label = {}

    set_user = set(user_labels)
    set_model = set([l for l in model_labels if l is not None])

    only_in_user = sorted(set_user - set_model)
    only_in_model = sorted(set_model - set_user)
    in_both = sorted(set_user & set_model)

    mismatched_ids = []
    for lab in in_both:
        uid = user_labels.index(lab)
        mid = model_label2id.get(lab)
        if mid is None or mid != uid:
            mismatched_ids.append((lab, uid, mid))

    print(f"user labels: {len(user_labels)}, model labels: {len(model_labels)}")
    print("Only in user (sample):", only_in_user[:30])
    print("Only in model (sample):", only_in_model[:30])
    if mismatched_ids:
        print("Labels with different ids (label, user_index, model_index):")
        for t in mismatched_ids[:50]:
            print(t)
    else:
        print("No differing id assignments for common labels.")
    return {
        "only_in_user": only_in_user,
        "only_in_model": only_in_model,
        "mismatched_ids": mismatched_ids,
        "user_len": len(user_labels),
        "model_len": len(model_labels),
    }


# Example usage (update paths to your files):
res = compare_label_lists("../outputs/unique_labels_old.json", "../models/NER_mudel_v2")
print(res)
