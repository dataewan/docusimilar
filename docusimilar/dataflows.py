import spacy
from dataclasses import dataclass
from typing import List
import tqdm
import os
import pickle

import logging

DOC_CACHE = "doc.pkl"

logging.basicConfig(level=logging.INFO)

nlp = spacy.load("en_core_web_sm")


@dataclass
class Document:
    idx: int
    path: str
    title: str
    sentences: List[str]


def parse_document(path: str, idx: int) -> Document:
    with open(path, "r") as f:
        text = f.read()

    lines = [line for line in text.split("\n") if line != ""]

    title = lines[0]
    contents = "".join(lines[1:])
    tokens = nlp(contents)
    sentences = [sent.string.strip() for sent in tokens.sents]

    return Document(idx=idx, path=path, title=title, sentences=sentences)


def parse_documents(documentpaths: List[str]) -> List[Document]:
    docs = []
    for idx, documentpath in enumerate(tqdm.tqdm(documentpaths)):
        docs.append(parse_document(documentpath, idx))

    return docs


def load_or_parse_documents(
    documentpaths: List[str], overwrite: bool = False
) -> List[Document]:
    if overwrite or not os.path.exists(DOC_CACHE):
        docs = parse_documents(documentpaths)
        with open(DOC_CACHE, "wb") as f:
            pickle.dump(docs, f)

        return docs

    else:
        with open(DOC_CACHE, "rb") as f:
            docs = pickle.load(f)

            return docs
