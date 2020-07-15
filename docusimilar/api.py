from typing import List
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from .dataflows import load_or_parse_documents, Document
from .similarity import load_or_calculate_embeddings, EmbeddingDescription
import glob
import os
import faiss
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates/")


def get_embeddings() -> List[EmbeddingDescription]:
    basepath = "/Users/ewannicolson/data/bbc/bbc/"
    files = glob.glob(os.path.join(basepath, "*/*.txt"))
    docs = load_or_parse_documents(files)
    embeddings = load_or_calculate_embeddings(docs)
    return embeddings


similarity_details = {}


@app.on_event("startup")
def setup():
    embeddings = get_embeddings()
    doc_lookup = {idx: e.doc for idx, e in enumerate(embeddings)}
    vectors = np.array([e.centre for e in embeddings])
    dimension = vectors[0].shape[0]

    similarity_details["doc_lookup"] = doc_lookup

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    _, neighbours = index.search(vectors, 5)

    similarity_details["neighbours"] = neighbours


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("main.html", context={"request": request})


def get_similar(item_id: int) -> List[Document]:
    neighbours = similarity_details["neighbours"][item_id]
    return [
        similarity_details["doc_lookup"][neighbour_id]
        for neighbour_id in neighbours[1:]
    ]


@app.get("/item/{item_id}")
async def root(request: Request, item_id: int):
    similar_documents = get_similar(item_id)
    document = similarity_details["doc_lookup"][item_id]
    return templates.TemplateResponse(
        "item.html",
        context={
            "request": request,
            "item_id": item_id,
            "similar_documents": similar_documents,
            "document": document,
        },
    )
