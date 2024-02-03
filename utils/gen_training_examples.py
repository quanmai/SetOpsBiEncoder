import argparse
import logging
import random
from utils.example_utils import (
    read_examples,
    read_documents,
    read_documents_flatten,
    Document,
    TrainingExample,
)

logger = logging.getLogger()

def gen_training_examples(
    fdocument: str,
    fexamples: str,
    fbm25: str,
    num_hard_negatives: int,
    num_random_negatives: int,
    output: str,
) -> None:
    documents = read_documents(fdocument)
    examples = read_examples(fexamples)
    bm25output = read_examples(fbm25)

    # Map document title to text
    doc_title_to_text = {doc.title: doc.text for doc in documents}
    outputs = []
    for example, predictions in zip(examples, bm25output):
        logger.info("Processing example %s." % example.query)
        relevant_titles = set(example.docs)
        if len(relevant_titles) == 0:
            raise ValueError(f"Missing relevant documents for query: `{example.query}`")
        
        # Add relevant documents as positive examples.
        pos_docs = [
            Document(
                title=doc_title,
                text=doc_title_to_text[doc_title],
            ) for doc_title in relevant_titles
        ]

        # Add BM25 negative examples as hard negatives.
        hard_neg_docs = []
        for doc_title in predictions.docs:
            if doc_title not in relevant_titles:
                if doc_title not in doc_title_to_text:
                    raise ValueError(f"Missing document title: `{doc_title}`")
                hard_neg_docs.append(
                    Document(
                        title=doc_title,
                        text=doc_title_to_text[doc_title],
                    )
                )
            if len(hard_neg_docs) >= num_hard_negatives:
                break    
    
        # Add random non-relevant examples as negative documents.
        neg_docs = []
        random_titles = random.sample(
            list(doc_title_to_text.keys()),
            k = 2 * num_random_negatives,
        )
        for doc_title in random_titles:
            if doc_title not in relevant_titles:
                neg_docs.append(
                    Document(
                        title=doc_title,
                        text=doc_title_to_text[doc_title],
                    )
                )
            if len(neg_docs) >= num_random_negatives:
                break
        outputs.append(
            TrainingExample(
                query=example.query,
                positive_docs=pos_docs,
                negative_docs=neg_docs,
                hard_negative_docs=hard_neg_docs,
            )
        )

    import jsonpickle
    output_pickpled = jsonpickle.encode(outputs, unpicklable=False)
    with open(output, "w") as json_file:
        json_file.write(output_pickpled)

    print(f"Data has been dump to {output}")

def gen_training_examples_flatten(
    fdocument: str, 
    fexamples: str,
    fbm25: str,
    num_hard_negatives: int,
    num_random_negatives: int,
    min_num_positives: int,
    output: str,
    ):

    documents = read_documents_flatten(fdocument)
    examples = read_examples(fexamples)
    bm25output = read_examples(fbm25)

    # Map document title to text
    doc_title_to_text = {doc["title"]: doc["text"] for doc in documents}
    outputs = []
    for example, predictions in zip(examples, bm25output):
        logger.info("Processing example %s." % example.query)
        relevant_titles = set(example.docs)
        if len(relevant_titles) == 0:
            # raise ValueError(f"Missing relevant documents for query: `{example.query}`")
            continue

        # Add relevant documents as positive examples.
        pos_docs = [
            {
                "title": doc_title,
                "text": doc_title_to_text[doc_title],
            } for doc_title in relevant_titles
        ]

        # 
        if len(pos_docs) < min_num_positives:
            pos_docs += [
                pos_docs[0] for _ in range(min_num_positives - len(pos_docs))
            ]

        # Add BM25 negative examples as hard negatives.
        hard_neg_docs = []
        for doc_title in predictions.docs:
            if doc_title not in relevant_titles:
                if doc_title not in doc_title_to_text:
                    raise ValueError(f"Missing document title: `{doc_title}`")
                hard_neg_docs.append(
                    {
                        "title": doc_title,
                        "text": doc_title_to_text[doc_title],
                    }
                )
            if len(hard_neg_docs) == num_hard_negatives:
                break    
        assert len(hard_neg_docs) == num_hard_negatives
        # Add random non-relevant examples as negative documents.
        neg_docs = []
        random_titles = random.sample(
            list(doc_title_to_text.keys()),
            k = 2 * num_random_negatives,
        )
        for doc_title in random_titles:
            if doc_title not in relevant_titles:
                neg_docs.append(
                    {
                        "title": doc_title,
                        "text": doc_title_to_text[doc_title],
                    }
                )
            if len(neg_docs) == num_random_negatives:
                break
        assert len(neg_docs) == num_random_negatives
        outputs.append(
            {
                "query": example.query,
                "original_query": example.original_query,
                "template": example.metadata.template,
                "positive_docs": pos_docs,
                "negative_docs": neg_docs,
                "hard_negative_docs": hard_neg_docs,
            }
        )

    import json
    with open(output, "w") as json_file:
        json_file.write(json.dumps(outputs, indent=4) + "\n")

    print(f"Data has been dump to {output}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fdocument",
        default="../quest/dataset/documents.jsonl",
        type=str,
    )
    parser.add_argument(
        "--fbm25",
        default="../quest/output/bm25output.jsonl",
        type=str,
    )
    parser.add_argument(
        "--num-hard-negatives",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--num-random-negatives",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gen-dev",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gen-test",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    output = "./data/training_example123.json"
    fexamples = "../quest/dataset/train.jsonl"
    if args.gen_dev:
        output = "./data/dev_example.json"
        fexamples = "../quest/dataset/val.jsonl"
    elif args.gen_test:
        output = "./data/test_example.json"
        fexamples = "../quest/dataset/test.jsonl"

    if args.flatten:
        gen_training_examples_flatten(
            fdocument=args.fdocument,
            fexamples=fexamples,
            fbm25=args.fbm25,
            num_hard_negatives=args.num_hard_negatives,
            num_random_negatives=args.num_random_negatives,
            min_num_positives=10,
            output=output,
        )
    else:
        gen_training_examples(
            fdocument=args.fdocument,
            fexamples=fexamples,
            fbm25=args.fbm25,
            num_hard_negatives=args.num_hard_negatives,
            num_random_negatives=args.num_random_negatives,
            output=output,
        )


if __name__ == "__main__":
    main()