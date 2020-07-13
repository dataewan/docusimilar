from .dataflows import Document
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from dataclasses import dataclass
import tqdm
import os
import pickle


embedder = SentenceTransformer("bert-base-nli-mean-tokens")

EMB_CACHE = "emb.pkl"


@dataclass
class EmbeddingDescription:
    doc: Document
    embeddings: List[np.ndarray]
    centre: np.ndarray


def calculate_embedding(doc: Document) -> EmbeddingDescription:
    embeddings = embedder.encode(doc.sentences)
    centre = np.mean(embeddings, axis=0)
    return EmbeddingDescription(doc=doc, embeddings=embeddings, centre=centre)


def calculate_embeddings(docs: List[Document]) -> List[EmbeddingDescription]:
    embeddings = []

    for doc in tqdm.tqdm(docs):
        embeddings.append(calculate_embedding(doc))

    return embeddings


def load_or_calculate_embeddings(
    docs: List[Document], overwrite: bool = False
) -> List[EmbeddingDescription]:
    if overwrite or not os.path.exists(EMB_CACHE):
        embeddings = calculate_embeddings(docs)
        with open(EMB_CACHE, "wb") as f:
            pickle.dump(embeddings, f)

        return embeddings

    else:
        with open(EMB_CACHE, "rb") as f:
            embeddings = pickle.load(f)

        return embeddings
