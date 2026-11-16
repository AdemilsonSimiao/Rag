import re
from typing import Dict, List, Protocol, Tuple
from error import fail
from src.models import MinimalSource


class TextGenerator(Protocol):
    """A function that turns chat messages into the model's reply."""

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        """Return the reply for the given chat messages."""


def read_span(file_path: str, first: int, last: int) -> str:
    """Read the characters first..last of a file, as the chunker saw them."""
    with open(file_path, encoding="utf-8", errors="replace") as file:
        return file.read()[first:last]


def build_context(
    passages: List[Tuple[MinimalSource, str]], max_chars: int
) -> str:
    """Join the passages, best first, stopping at a character budget."""
    separator = "\n\n"
    blocks: List[str] = []
    remaining: int = max_chars
    for number, (source, text) in enumerate(passages, start=1):
        header = (
            f"[{number}] {source.file_path} "
            f"(characters {source.first_character_index}"
            f"-{source.last_character_index})\n"
        )
        cost = len(header) + (len(separator) if blocks else 0)
        if remaining <= cost:
            break
        body = text[: remaining - cost]
        blocks.append(header + body)
        remaining -= cost + len(body)
    return separator.join(blocks)


def build_messages(question: str, context: str) -> List[Dict[str, str]]:
    """Build the chat prompt: rules for the model, context and question."""
    system = (
        "You answer questions about the vLLM codebase and documentation. "
        "Use ONLY the context provided. If the context does not contain "
        "the answer, say that you cannot find it in the provided sources. "
        "Be concise and precise, and copy identifiers, file names and "
        "values exactly as they appear in the context."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def clean_answer(raw: str) -> str:
    """Remove any <think> block the model may have written."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    return raw.split("<think>")[0].strip()


def answer_question(
    generate: TextGenerator,
    question: str,
    passages: List[Tuple[MinimalSource, str]],
    max_chars: int = 8000,
) -> str:
    """Answer one question from its passages; never calls the model blind."""
    if not passages:
        return "No relevant sources were found for this question."
    context = build_context(passages, max_chars)
    reply = clean_answer(generate(build_messages(question, context)))
    return reply or "The model did not produce an answer."


def load_model(
    model_name: str = "Qwen/Qwen3-0.6B", max_new_tokens: int = 256
) -> TextGenerator:
    """Load the model once and return a greedy text generation function."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    def generate(messages: List[Dict[str, str]]) -> str:
        """Run the model on the chat messages and return only the reply."""
        prompt: str = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        reply_tokens = output[0][inputs["input_ids"].shape[1]:]
        return str(tokenizer.decode(reply_tokens, skip_special_tokens=True))

    return generate

def load_generator(model_name: str) -> TextGenerator:
    """Load the model, or exit with a clear message if it cannot load."""
    try:
        return load_model(model_name)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        fail(f"cannot load the model {model_name}: {error}")