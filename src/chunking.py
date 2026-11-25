import re

from .ingestion import load_documents


def extract_sections(text):
    """
    Split document text into numbered policy sections.

    Example:

    3. Attendance and Semester Eligibility
    <content>

    becomes:

    {
        "section": "3. Attendance and Semester Eligibility",
        "text": "<content>"
    }
    """

    pattern = r"(?m)^(?P<section>\d+\.\s+[^\n]+)\n"

    matches = list(re.finditer(pattern, text))

    sections = []

    if not matches:
        return [
            {
                "section": "General",
                "text": text.strip(),
            }
        ]

    # Text before the first numbered section
    if matches[0].start() > 0:
        preamble = text[:matches[0].start()].strip()

        if preamble:
            sections.append(
                {
                    "section": "Document Information",
                    "text": preamble,
                }
            )

    for i, match in enumerate(matches):

        section_name = match.group("section").strip()

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        section_text = text[start:end].strip()

        if section_text:
            sections.append(
                {
                    "section": section_name,
                    "text": section_text,
                }
            )

    return sections


def chunk_text(text, chunk_size=800, overlap=100):
    """
    Split long section text into overlapping chunks.

    The chunking is character-based for now.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size - overlap

    return chunks


def create_chunks():

    documents = load_documents()

    chunks = []

    for document in documents:

        sections = extract_sections(document["text"])

        for section in sections:

            if section["section"] == "Document Information":
                continue

            section_chunks = chunk_text(section["text"])

            for chunk in section_chunks:

                chunks.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "source": document["metadata"]["source"],
                            "page": document["metadata"]["page"],
                            "section": section["section"],
                        },
                    }
                )

    return chunks


if __name__ == "__main__":

    chunks = create_chunks()

    print(f"Created {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks[:10], start=1):

        print("=" * 80)
        print(f"CHUNK {i}")
        print("=" * 80)

        print("Source :", chunk["metadata"]["source"])
        print("Page   :", chunk["metadata"]["page"])
        print("Section:", chunk["metadata"]["section"])

        print("\nText:")
        print(chunk["text"])