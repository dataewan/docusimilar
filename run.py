import glob
from docusimilar import dataflows, similarity
import os


def run():
    basepath = "/Users/ewannicolson/data/bbc/bbc/"
    files = glob.glob(os.path.join(basepath, "*/*.txt"))
    docs = dataflows.load_or_parse_documents(files)
    embeddings = similarity.load_or_calculate_embeddings(docs)
    return embeddings


if __name__ == "__main__":
    embeddings = run()
