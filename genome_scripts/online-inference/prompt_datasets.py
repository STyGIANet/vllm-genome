from __future__ import annotations

from typing import Any, Callable

DatasetFormatter = Callable[[dict[str, Any]], dict[str, Any] | None]
DatasetSpec = tuple[str, str | None, DatasetFormatter, str]


def format_hotpotqa(ex):
    try:
        sentences = ex.get("context", {}).get("sentences", [])
        flat = []

        for para in sentences:
            flat.extend(para)

        context = " ".join(flat[:50])
        question = ex.get("question", "")

        return {
            "question": f"{context}\n\nQuestion: {question}",
            "choices": ["Yes", "No"],
            "answer": 0,
        }
    except Exception:
        return None


def format_pubmedqa(ex):
    try:
        contexts = ex.get("context", {}).get("contexts", [])
        context = " ".join(contexts[:5])

        question = ex.get("question", "")

        choices = ["yes", "no", "maybe"]
        answer_map = {"yes": 0, "no": 1, "maybe": 2}

        answer = answer_map.get(ex.get("final_decision", ""), 0)

        return {
            "question": f"{context}\n\nQuestion: {question}",
            "choices": choices,
            "answer": answer,
        }
    except Exception:
        return None


def format_arc(ex):
    return {
        "question": ex["question"],
        "choices": ex["choices"]["text"],
        "answer": ord(ex["answerKey"]) - ord("A"),
    }


def format_openbookqa(ex):
    return {
        "question": ex["question_stem"],
        "choices": ex["choices"]["text"],
        "answer": ord(ex["answerKey"]) - ord("A"),
    }


def format_csqa(ex):
    return {
        "question": ex["question"],
        "choices": ex["choices"]["text"],
        "answer": ord(ex["answerKey"]) - ord("A"),
    }


def format_boolq(ex):
    return {
        "question": ex["question"],
        "choices": ["True", "False"],
        "answer": 0 if ex["answer"] else 1,
    }


def format_piqa(ex):
    return {
        "question": ex["goal"],
        "choices": [ex["sol1"], ex["sol2"]],
        "answer": ex["label"],
    }


def format_mmlu(ex):
    answer = ex["answer"]
    if isinstance(answer, str):
        answer = ord(answer.strip().upper()) - ord("A")

    return {
        "question": ex["question"],
        "choices": list(ex["choices"]),
        "answer": int(answer),
    }


ALL_DATASETS: dict[str, DatasetSpec] = {
    "mmlu_abstract_algebra": (
        "cais/mmlu",
        "abstract_algebra",
        format_mmlu,
        "test",
    ),
    "hotpot_qa_fullwiki": (
        "hotpot_qa",
        "fullwiki",
        format_hotpotqa,
        "validation",
    ),
    "pubmed_qa_pqa_labeled": (
        "pubmed_qa",
        "pqa_labeled",
        format_pubmedqa,
        "train",
    ),
    "arc_challenge": (
        "ai2_arc",
        "ARC-Challenge",
        format_arc,
        "validation",
    ),
    "arc_easy": (
        "ai2_arc",
        "ARC-Easy",
        format_arc,
        "validation",
    ),
    "openbookqa_main": (
        "openbookqa",
        "main",
        format_openbookqa,
        "validation",
    ),
    "commonsense_qa": (
        "commonsense_qa",
        None,
        format_csqa,
        "validation",
    ),
    "boolq": (
        "boolq",
        None,
        format_boolq,
        "validation",
    ),
    "piqa": (
        "piqa",
        None,
        format_piqa,
        "validation",
    ),
}


SEND_PROMPTS_DATASETS: list[DatasetSpec] = [
    ALL_DATASETS["hotpot_qa_fullwiki"],
    # ALL_DATASETS["mmlu_abstract_algebra"],
    # ALL_DATASETS["pubmed_qa_pqa_labeled"],
    # ALL_DATASETS["arc_challenge"],
    # ALL_DATASETS["arc_easy"],
    # ALL_DATASETS["openbookqa_main"],
    # ALL_DATASETS["commonsense_qa"],
    # ALL_DATASETS["boolq"],
    # ALL_DATASETS["piqa"],
]


WEIGHTS_EXPERIMENT_DATASETS: list[DatasetSpec] = [
    ALL_DATASETS["mmlu_abstract_algebra"],
    ALL_DATASETS["hotpot_qa_fullwiki"],
    ALL_DATASETS["pubmed_qa_pqa_labeled"],
    ALL_DATASETS["arc_challenge"],
    ALL_DATASETS["arc_easy"],
    ALL_DATASETS["openbookqa_main"],
    ALL_DATASETS["commonsense_qa"],
    ALL_DATASETS["boolq"],
    ALL_DATASETS["piqa"],
]
