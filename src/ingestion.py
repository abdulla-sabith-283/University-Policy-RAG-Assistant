from pathlib import Path
import pymupdf


DOCUMENTS_DIR = Path("data/documents")


def load_documents():
    documents = []

    for pdf_path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        pdf = pymupdf.open(pdf_path)

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                }
            )

        pdf.close()

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print(f"Loaded {len(documents)} pages\n")

    for document in documents[:2]:
        print("=" * 80)
        print("SOURCE:", document["metadata"]["source"])
        print("PAGE:", document["metadata"]["page"])
        print("=" * 80)
        print(document["text"][:1000])