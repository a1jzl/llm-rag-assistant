"""Generation de reponses sourcees a partir des chunks recuperes.

Ce module construit un prompt combinant la question de l'utilisateur
et les chunks pertinents recuperes par le Retriever, puis interroge
un LLM pour produire une reponse ancree dans le corpus documentaire.
"""

import logging
import os

from openai import OpenAI

from src.retriever import Retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Tu es un assistant qui repond aux questions uniquement a partir du "
    "contexte fourni. Si l'information n'est pas presente dans le contexte, "
    "indique clairement que tu ne peux pas repondre avec certitude."
)


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "

".join(f"[Extrait {i+1}] {c}" for i, c in enumerate(contexts))
    return (
        f"Contexte:
{context_block}

"
        f"Question: {question}

"
        "Reponds de facon concise en citant les extraits utilises entre crochets."
    )


class RagAssistant:
    """Assistant conversationnel combinant retrieval et generation."""

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def answer(self, question: str, model: str = "gpt-4o-mini") -> str:
        contexts = self.retriever.search(question)
        prompt = build_prompt(question, contexts)

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        logger.info("Question: %s | Reponse generee (%d caracteres)", question, len(answer))
        return answer


if __name__ == "__main__":
    assistant = RagAssistant()
    print(assistant.answer("Quelle est la procedure de remboursement ?"))
