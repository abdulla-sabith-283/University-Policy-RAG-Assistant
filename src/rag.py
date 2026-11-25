import re

from .retriever import Retriever
from .reranker import CrossEncoderReranker
from .generator import generate_answer


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = CrossEncoderReranker()

    def is_suspicious(self, question):

        suspicious_patterns = [
            r"ignore\s+(all\s+)?previous\s+instructions",
            r"ignore\s+(all\s+)?prior\s+instructions",
            r"reveal\s+(your\s+)?system\s+(prompt|instructions)",
            r"print\s+(your\s+)?system\s+(prompt|instructions)",
            r"show\s+(your\s+)?system\s+(prompt|instructions)",
            r"what\s+(are|is)\s+your\s+system\s+(prompt|instructions)",
            r"reveal\s+your\s+instructions",
            r"print\s+your\s+instructions",
            r"show\s+your\s+instructions",
        ]

        question_lower = question.lower()

        for pattern in suspicious_patterns:
            if re.search(pattern, question_lower):
                return True

        return False

    def is_greeting(self, question):

        greeting_patterns = [
            r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings|howdy)\s*[!?.]*\s*$",
        ]

        question_lower = question.lower().strip()

        for pattern in greeting_patterns:
            if re.search(pattern, question_lower):
                return True

        return False

    def build_context(self, results):

        context_parts = []

        for i, result in enumerate(results, start=1):

            metadata = result["metadata"]

            context_parts.append(
                f"""
[EVIDENCE {i}]
Source: {metadata["source"]}
Page: {metadata["page"]}
Section: {metadata["section"]}

{result["text"]}
"""
            )

        return "\n".join(context_parts)

    def answer(self, question):

        # -----------------------------
        # 1. SECURITY CHECK
        # -----------------------------

        if self.is_suspicious(question):

            return {
                "answer": (
                    "I can't provide internal instructions "
                    "or system configuration. I can help answer "
                    "questions about the provided university policies."
                ),
                "sources": [],
                "evidence": [],
            }

        # -----------------------------
        # 1B. GREETING CHECK
        # -----------------------------

        if self.is_greeting(question):

            return {
                "answer": (
                    "Hello! I'm the Northbridge Policy Assistant. "
                    "Ask me about university policies — attendance, "
                    "library, examinations, scholarships, and more."
                ),
                "sources": [],
                "evidence": [],
            }

        # -----------------------------
        # 2. RETRIEVAL
        # -----------------------------

        retrieval = self.retriever.retrieve_with_gate(
            question,
            top_k=5,
        )

        results = retrieval["results"]

        # -----------------------------
        # 3. RELEVANCE GATE
        # -----------------------------

        if not retrieval["relevant"]:

            return {
                "answer": (
                    "I could not find sufficient information "
                    "in the provided documents."
                ),
                "sources": [],
                "evidence": [],
            }

        # -----------------------------
        # 4. BUILD CONTEXT
        # -----------------------------

        context = self.build_context(results)

        # -----------------------------
        # 5. GENERATE
        # -----------------------------

        answer = generate_answer(
            question,
            context,
        )

        # -----------------------------
        # 6. FIND CITED EVIDENCE
        # -----------------------------

        cited_numbers = set(
            int(number)
            for number in re.findall(
                r"\[EVIDENCE\s+(\d+)\]",
                answer,
            )
        )

        cited_evidence = []

        for number in sorted(cited_numbers):

            if 1 <= number <= len(results):

                result = results[number - 1]

                metadata = result["metadata"]

                cited_evidence.append(
                    {
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "section": metadata["section"],
                        "text": result["text"],
                    }
                )

        # -----------------------------
        # 7. CLEAN ANSWER
        # -----------------------------

        clean_answer = re.sub(
            r"\s*\[EVIDENCE\s+\d+\]",
            "",
            answer,
        )

        clean_answer = re.sub(
            r"\n+",
            "\n",
            clean_answer,
        ).strip()

        # -----------------------------
        # 8. RETURN RESULT
        # -----------------------------

        return {
            "answer": clean_answer,
            "sources": cited_evidence,
            "evidence": results,
        }

    def answer_hybrid(self, question):

        # -----------------------------
        # 1. SECURITY CHECK
        # -----------------------------

        if self.is_suspicious(question):

            return {
                "answer": (
                    "I can't provide internal instructions "
                    "or system configuration. I can help answer "
                    "questions about the provided university policies."
                ),
                "sources": [],
                "evidence": [],
            }

        # -----------------------------
        # 1B. GREETING CHECK
        # -----------------------------

        if self.is_greeting(question):

            return {
                "answer": (
                    "Hello! I'm the Northbridge Policy Assistant. "
                    "Ask me about university policies — attendance, "
                    "library, examinations, scholarships, and more."
                ),
                "sources": [],
                "evidence": [],
            }

        # -----------------------------
        # 2. HYBRID RETRIEVAL
        # -----------------------------

        hybrid_results = self.retriever.retrieve_hybrid(
            question,
            top_k=10,
        )

        # -----------------------------
        # 3. RERANK
        # -----------------------------

        reranked = self.reranker.rerank(
            question,
            hybrid_results,
            top_n=5,
        )

        # -----------------------------
        # 4. RELEVANCE GATE
        # -----------------------------

        if not reranked:

            return {
                "answer": (
                    "I could not find sufficient information "
                    "in the provided documents."
                ),
                "sources": [],
                "evidence": [],
            }

        # -----------------------------
        # 5. BUILD CONTEXT
        # -----------------------------

        context = self.build_context(reranked)

        # -----------------------------
        # 6. GENERATE
        # -----------------------------

        answer = generate_answer(
            question,
            context,
        )

        # -----------------------------
        # 7. FIND CITED EVIDENCE
        # -----------------------------

        cited_numbers = set(
            int(number)
            for number in re.findall(
                r"\[EVIDENCE\s+(\d+)\]",
                answer,
            )
        )

        cited_evidence = []

        for number in sorted(cited_numbers):

            if 1 <= number <= len(reranked):

                result = reranked[number - 1]

                metadata = result["metadata"]

                cited_evidence.append(
                    {
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "section": metadata["section"],
                        "text": result["text"],
                    }
                )

        # -----------------------------
        # 8. CLEAN ANSWER
        # -----------------------------

        clean_answer = re.sub(
            r"\s*\[EVIDENCE\s+\d+\]",
            "",
            answer,
        )

        clean_answer = re.sub(
            r"\n+",
            "\n",
            clean_answer,
        ).strip()

        # -----------------------------
        # 9. RETURN RESULT
        # -----------------------------

        return {
            "answer": clean_answer,
            "sources": cited_evidence,
            "evidence": reranked,
        }


if __name__ == "__main__":

    rag = RAGPipeline()

    print("=" * 80)
    print("NORTHBRIDGE POLICY RAG")
    print("=" * 80)
    print("Type 'exit' to quit.")

    while True:

        question = input("\nQuestion: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            break

        result = rag.answer(question)

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)

        print(result["answer"])

        if result["sources"]:

            print("\n" + "=" * 80)
            print("SOURCES")
            print("=" * 80)

            for source in result["sources"]:

                print(
                    f"- {source['source']} "
                    f"| Page {source['page']} "
                    f"| {source['section']}"
                )