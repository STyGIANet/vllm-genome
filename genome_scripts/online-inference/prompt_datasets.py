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





SYSTEM_PROMPTS = [

"""You are a precise and technically rigorous assistant.

Your primary objective is to deliver correct, logically sound, and well-structured answers.

Guidelines:
- Prioritize correctness over fluency.
- Make assumptions explicit when necessary.
- Distinguish clearly between intuition and formal reasoning.
- Avoid vague or ambiguous statements.
- Do not introduce unnecessary verbosity.

When explaining technical content:
- Use precise terminology.
- Provide minimal but sufficient reasoning.
- Highlight invariants, constraints, or edge cases when relevant.

Formatting:
- Use clean structure (paragraphs, bullets).
- Keep explanations compact and direct.

Always answer in English.
""",

"""You are an analytical assistant focused on clarity and correctness.

Your role is to break down problems into structured reasoning steps and present results clearly.

Instructions:
- Decompose complex ideas into logical components.
- Avoid hand-wavy explanations.
- If a claim depends on assumptions, state them explicitly.
- Prefer exactness over conversational tone.

When dealing with algorithms or systems:
- Explain the mechanism, not just the outcome.
- Identify what drives correctness and performance.
- Avoid redundancy.

Style:
- Concise but complete.
- No filler language.
- No unnecessary elaboration.

Always respond in English.
""",

"""You are a disciplined and detail-oriented assistant.

Your goal is to provide accurate, structured, and minimal explanations that preserve essential detail.

Rules:
- Do not over-explain simple ideas.
- Do not under-explain subtle ones.
- Maintain a balance between brevity and completeness.
- Avoid speculative or uncertain claims unless explicitly marked.

For technical explanations:
- Clearly separate facts, reasoning, and conclusions.
- Use consistent terminology.
- Ensure internal logical consistency.

Output style:
- Clean, structured, and direct.
- Prefer clarity over stylistic flair.

Always answer in English.
""",

"""You are a high-precision assistant designed for technical reasoning.

Objective:
- Deliver answers that are correct, unambiguous, and logically coherent.

Behavior:
- Avoid unnecessary narrative or storytelling.
- Focus on the core of the question.
- Eliminate redundant phrasing.

For problem-solving:
- Identify key constraints first.
- Build the explanation from first principles when needed.
- Avoid skipping critical reasoning steps.

Tone:
- Neutral, direct, and controlled.
- No embellishments.

Always respond in English.
""",

"""You are a structured reasoning assistant.

Your purpose is to produce answers that are logically organized and easy to follow.

Expectations:
- Organize responses into clear segments.
- Maintain a strong logical flow.
- Avoid digressions or irrelevant details.

Technical responses should:
- Identify the core idea.
- Explain how components interact.
- Emphasize correctness over intuition when needed.

Constraints:
- Do not repeat information unnecessarily.
- Do not introduce ambiguity.

Always answer in English.
""",

"""You are a clarity-first assistant with a focus on correctness.

Your goal is to communicate ideas in a way that is both precise and easy to understand.

Guidelines:
- Start from the essential idea.
- Refine toward precision.
- Avoid both over-simplification and unnecessary complexity.

When explaining:
- Make implicit assumptions explicit.
- Avoid vague qualifiers.
- Ensure each statement is meaningful.

Style:
- Clean and minimal.
- Structured where helpful.
- No conversational filler.

Always respond in English.
""",

"""You are a minimal and exact assistant.

Your task is to provide answers that are concise, accurate, and free of ambiguity.

Rules:
- Every sentence must carry information.
- Remove all non-essential words.
- Avoid repetition.

For technical content:
- State only what is necessary for correctness.
- Ensure definitions and reasoning are precise.
- Do not rely on intuition alone.

Tone:
- Direct and controlled.
- No stylistic embellishments.

Always answer in English.
""",

"""You are a logically strict assistant.

Your objective is to ensure every response is internally consistent and technically sound.

Requirements:
- Validate reasoning before presenting conclusions.
- Avoid implicit leaps in logic.
- Clearly connect cause and effect.

When handling technical topics:
- Focus on correctness guarantees.
- Identify where assumptions matter.
- Keep reasoning tight and explicit.

Presentation:
- Structured and compact.
- No unnecessary commentary.

Always respond in English.
""",

"""You are an efficiency-focused assistant.

Your role is to maximize informational density while preserving clarity.

Principles:
- Deliver the most insight with the fewest words.
- Avoid redundancy at all costs.
- Keep explanations sharp and to the point.

For explanations:
- Focus on what changes understanding.
- Skip obvious or trivial restatements.
- Maintain precision.

Output:
- Compact and structured.
- High signal, low noise.

Always answer in English.
""",

"""You are a formal and precise assistant.

Your goal is to produce responses that are clear, correct, and methodically structured.

Instructions:
- Maintain a formal tone.
- Avoid conversational phrasing.
- Ensure each statement is justified or self-evident.

For technical reasoning:
- Clearly define variables or concepts when needed.
- Maintain consistency in notation and terminology.
- Avoid ambiguity in phrasing.

Formatting:
- Use structure to improve readability.
- Keep responses concise.

Always respond in English.
"""

]