from rag.logging_config import configure_logging
import logging

configure_logging()
logger = logging.getLogger(__name__)

def grounded_qa_prompt(context: str, question: str) -> str:
    #logger.info(f"Question : {question}")
    #logger.info(f"Context : {context}")
    return (
        "You are a careful assistant.\n\n"
        "Extract the exact sentence(s) from the context that answer the question.\n"
        "If the answer is not present, say: \"Not found in the provided documents.\".\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )
