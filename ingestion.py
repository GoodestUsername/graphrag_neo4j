import re
from typing import List

from store import Section


def chunk_text_whitespace_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    index = 0

    while len(text) > index:
        prev_whitespace = 0
        left_index = index - overlap
        while left_index >= 0:
            if text[left_index] == " ":
                prev_whitespace = left_index
                break
            left_index -= 1
        next_whitespace = text.find(" ", index + chunk_size)
        if next_whitespace == -1:
            next_whitespace = len(text)
        chunk = text[prev_whitespace:next_whitespace].strip()
        chunks.append(chunk)
        index = next_whitespace + 1
    return chunks


def chunk_text_size_split(text: str, chunk_size: int, overlap: int):
    chunks = []
    index = 0

    while len(text) > index:
        start = max(0, index - overlap + 1)
        end = min(index + chunk_size + overlap, len(text))
        chunk = text[start:end].strip()
        chunks.append(chunk)
        index += chunk_size
    return chunks


def chunk_text(
    text: str, chunk_size: int, overlap: int, split_on_whitespace_only=True
) -> List[str]:
    return (
        chunk_text_whitespace_split(text, chunk_size, overlap)
        if split_on_whitespace_only
        else chunk_text_size_split(text, chunk_size, overlap)
    )


def split_text_to_section_by_titles(text: str) -> List[Section]:
    title_pattern = re.compile(
        r"^(CHAPTER\s+\d+\.\s+.+?[.!?]|Epilogue|Prologue)$", re.DOTALL | re.MULTILINE
    )

    titles = [title.strip().lower() for title in title_pattern.findall(text)]
    sections = list(
        filter(lambda text: bool(text.strip()), re.split(title_pattern, text))
    )

    return [
        Section(id=title, text=title + sections[1 + i * 2])
        for i, title in enumerate(titles)
    ]
