import re
from typing import List, Dict, Any


def _extract_relevant_tokens(question: str, answer: str) -> set[str]:
    relevant_text = f"{question} {answer}".lower()
    return set(re.findall(r"\w+", relevant_text))


def compute_hit_at_k(retrieved_items: List[Dict[str, Any]], question: str, answer: str, k: int) -> float:
    """Return 1.0 if any of the top-k retrieved items is relevant to the question/answer pair."""
    relevant_tokens = _extract_relevant_tokens(question, answer)
    if not relevant_tokens:
        return 0.0

    top_items = retrieved_items[:k]
    for item in top_items:
        text = str(item.get("text", "")).lower()
        tokens = set(re.findall(r"\w+", text))
        if relevant_tokens & tokens:
            return 1.0
    return 0.0


def compute_hit_at_1(retrieved_items: List[Dict[str, Any]], question: str, answer: str) -> float:
    return compute_hit_at_k(retrieved_items, question, answer, 1)


def compute_hit_at_3(retrieved_items: List[Dict[str, Any]], question: str, answer: str) -> float:
    return compute_hit_at_k(retrieved_items, question, answer, 3)


def compute_hit_at_5(retrieved_items: List[Dict[str, Any]], question: str, answer: str) -> float:
    return compute_hit_at_k(retrieved_items, question, answer, 5)


def compute_context_precision(retrieved_items: List[Dict[str, Any]], question: str, answer: str, k: int = 5) -> float:
    """Compute context precision@k as the fraction of relevant items among the top-k retrieved chunks."""
    relevant_tokens = _extract_relevant_tokens(question, answer)
    if not relevant_tokens:
        return 0.0

    precision_scores = []
    for item in retrieved_items[:k]:
        text = str(item.get("text", "")).lower()
        tokens = set(re.findall(r"\w+", text))
        if relevant_tokens & tokens:
            precision_scores.append(1.0)
        else:
            precision_scores.append(0.0)

    return sum(precision_scores) / len(precision_scores) if precision_scores else 0.0


def compute_context_recall(retrieved_items: List[Dict[str, Any]], question: str, answer: str, k: int = 5) -> float:
    """Approximate context recall@k by checking whether the retrieved context covers the answer tokens."""
    answer_tokens = set(re.findall(r"\w+", answer.lower()))
    if not answer_tokens:
        return 0.0

    retrieved_text = " ".join(str(item.get("text", "")) for item in retrieved_items[:k]).lower()
    retrieved_tokens = set(re.findall(r"\w+", retrieved_text))
    if not retrieved_tokens:
        return 0.0

    overlap = answer_tokens & retrieved_tokens
    return len(overlap) / len(answer_tokens) if answer_tokens else 0.0
