from pathlib import Path
from typing import List, Dict


def ingest_txt_folder(folder_path: str, encoding: str = "utf-8") -> List[Dict[str, str]]:
    """
    Read all .txt files in a folder recursively.

    Returns:
        List of dicts:
        {
            "source": full file path,
            "filename": file name,
            "content": text content
        }
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    documents = []

    for file_path in folder.rglob("*.txt"):
        try:
            text = file_path.read_text(encoding=encoding)
            documents.append({
                "source": str(file_path),
                "filename": file_path.name,
                "content": text
            })
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    return documents


if __name__ == "__main__":
    docs = ingest_txt_folder("./data")

    print(f"Ingested {len(docs)} txt files.")
    for doc in docs[:3]:
        print("=" * 60)
        print(f"File: {doc['filename']}")
        print(f"Path: {doc['source']}")
        print(doc["content"][:300])