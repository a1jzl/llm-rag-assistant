"""Evaluation automatique de la qualite du pipeline RAG avec RAGAS.

Ce script prend un jeu de questions/reponses annote manuellement,
interroge l'assistant, puis calcule des metriques de pertinence du
contexte recupere et de fidelite de la reponse par rapport aux sources.
"""

import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.generate import RagAssistant

logger = logging.getLogger(__name__)

EVAL_SET_PATH = Path("evaluation/qa_testset.json")


def load_testset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation() -> None:
    testset = load_testset(EVAL_SET_PATH)
    assistant = RagAssistant()

    records = []
    for item in testset:
        contexts = assistant.retriever.search(item["question"])
        answer = assistant.answer(item["question"])
        records.append(
            {
                "question": item["question"],
                "contexts": contexts,
                "answer": answer,
                "ground_truth": item["reference_answer"],
            }
        )

    dataset = Dataset.from_list(records)
    results = evaluate(
        dataset,
        metrics=[context_precision, faithfulness, answer_relevancy],
    )
    logger.info("Resultats d'evaluation: %s", results)
    print(results)


if __name__ == "__main__":
    run_evaluation()
