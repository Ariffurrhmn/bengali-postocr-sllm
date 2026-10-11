"""Splits OCR text into ~200-300 subword-token chunks before correction,
following the base paper's (Kanerva et al. 2025) approach rather than
feeding a whole page at once — a whole page can overwhelm a 1B model and
produce degenerate output; shorter chunks give the model less to lose track
of and make individual failures easier to localize."""

DEFAULT_CHUNK_TOKENS = 250


def chunk_text(text: str, tokenizer, chunk_tokens: int = DEFAULT_CHUNK_TOKENS):
    """Splits on line breaks first (keeps natural page-line boundaries intact),
    then groups consecutive lines into chunks of roughly chunk_tokens tokens
    each, so a chunk boundary doesn't fall in the middle of a line."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return [text]

    chunks = []
    current_lines = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(tokenizer.encode(line, add_special_tokens=False))
        if current_lines and current_tokens + line_tokens > chunk_tokens:
            chunks.append("\n".join(current_lines))
            current_lines = []
            current_tokens = 0
        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


DEFAULT_CHUNK_CHARS = 300


def chunk_text_chars(text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS):
    """Model-independent alternative (review item M3): group complete lines
    into chunks of at most chunk_chars characters, so every model sees the
    same chunks whatever its tokenizer. A single longer line is its own chunk."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return [text]
    chunks, current, size = [], [], 0
    for line in lines:
        if current and size + len(line) + 1 > chunk_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    chunks.append("\n".join(current))
    return chunks
