import argparse
from utils.example_utils import read_documents

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fdocument",
        default="../quest/dataset/documents.jsonl",
        type=str,
    )

    args = parser.parse_args()
    documents = read_documents(args.fdocument)
    doc_2_idx = {doc.title: idx for idx, doc in enumerate(documents)}
    idx_2_doc = {idx: doc.title for idx, doc in enumerate(documents)}     
    
