import json
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from dotenv import load_dotenv
import requests

load_dotenv()

PROFILE_PATH = Path("data/candidate_profile/profile.json")
COLLECTION_NAME = "navneeth_candidate_profile"


class LMStudioEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        self.model = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")

    def __call__(self, input: Documents) -> Embeddings:
        payload = {
            "model": self.model,
            "input": list(input),
        }

        response = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            timeout=120,
        )

        response.raise_for_status()
        data = response.json()

        return [item["embedding"] for item in data["data"]]

def load_profile() -> Dict[str, Any]:
    with open(PROFILE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def flatten_profile(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate = profile["candidate"]
    docs = []

    for exp in candidate.get("experience", []):
        text = f"""
Type: Work Experience
Title: {exp["title"]}
Organization: {exp["organization"]}
Summary: {exp["summary"]}
Skills: {", ".join(exp.get("skills", []))}
CV Bullets: {" ".join(exp.get("cv_bullets", []))}
"""
        docs.append(
            {
                "id": exp["id"],
                "text": text,
                "metadata": {
                    "type": exp["type"],
                    "title": exp["title"],
                    "source_id": exp["id"],
                },
            }
        )

    for project in candidate.get("projects", []):
        text = f"""
Type: Project
Title: {project["title"]}
Summary: {project["summary"]}
Skills: {", ".join(project.get("skills", []))}
CV Bullets: {" ".join(project.get("cv_bullets", []))}
"""
        docs.append(
            {
                "id": project["id"],
                "text": text,
                "metadata": {
                    "type": project["type"],
                    "title": project["title"],
                    "source_id": project["id"],
                },
            }
        )

    for achievement in candidate.get("achievements", []):
        text = f"""
Type: Achievement
Title: {achievement["title"]}
Summary: {achievement["summary"]}
"""
        docs.append(
            {
                "id": achievement["id"],
                "text": text,
                "metadata": {
                    "type": "achievement",
                    "title": achievement["title"],
                    "source_id": achievement["id"],
                },
            }
        )

    skills_text = json.dumps(candidate.get("skills", {}), indent=2)
    docs.append(
        {
            "id": "skills_summary",
            "text": f"Type: Skills Summary\n{skills_text}",
            "metadata": {
                "type": "skills",
                "title": "Skills Summary",
                "source_id": "skills_summary",
            },
        }
    )

    education_text = json.dumps(candidate.get("education", []), indent=2)
    docs.append(
        {
            "id": "education_summary",
            "text": f"Type: Education Summary\n{education_text}",
            "metadata": {
                "type": "education",
                "title": "Education Summary",
                "source_id": "education_summary",
            },
        }
    )

    return docs


def get_chroma_collection(reset: bool = False):
    client = chromadb.PersistentClient(path="vector_store")
    embedding_function = LMStudioEmbeddingFunction()

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )


def ingest_profile(reset: bool = True) -> int:
    profile = load_profile()
    docs = flatten_profile(profile)
    collection = get_chroma_collection(reset=reset)

    collection.add(
        ids=[doc["id"] for doc in docs],
        documents=[doc["text"] for doc in docs],
        metadatas=[doc["metadata"] for doc in docs],
    )

    return len(docs)


def retrieve_relevant_evidence(job_description: str, top_k: int = 8) -> List[Dict[str, Any]]:
    collection = get_chroma_collection(reset=False)

    results = collection.query(
        query_texts=[job_description],
        n_results=top_k,
    )

    retrieved = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved


def format_evidence_for_prompt(evidence: List[Dict[str, Any]]) -> str:
    formatted = []

    for idx, item in enumerate(evidence, start=1):
        metadata = item["metadata"]

        formatted.append(
            f"""
Evidence {idx}
Title: {metadata.get("title")}
Type: {metadata.get("type")}
Source ID: {metadata.get("source_id")}
Retrieval distance: {item.get("distance")}

Content:
{item["text"]}
"""
        )

    return "\n---\n".join(formatted)