from dataclasses import dataclass
import jsonlines
import json
import typing
from typing import List, Optional, Dict


@dataclass
class ExampleMetadata:
    """Optional metadata used for analysis."""
    # The template used to synthesize this example.
    template: Optional[str] = None
    # The domain of the example (e.g. films, books, plants, animals).
    domain: Optional[str] = None
    # Fluency labels
    fluency: Optional[typing.Sequence[bool]] = None
    # Meaning labels
    meaning: Optional[typing.Sequence[bool]] = None
    # Naturalness labels
    naturalness: Optional[typing.Sequence[bool]] = None
    # The following fields are dictionaries keyed by document title.
    # The sequences can contain multiple values for replicated annotations.
    relevance_ratings: Optional[typing.Dict[str, typing.Sequence[str]]] = None
    evidence_ratings: Optional[typing.Dict[str, typing.Sequence[str]]] = None
    # The nested value is a map from query substring to document substring.
    attributions: Optional[typing.Dict[str, typing.Sequence[typing.Dict[str, str]]]] = None

@dataclass
class Example:
    """Represents a query paired with a set of documents."""
    query: str
    docs: typing.Iterable[str]
    original_query: Optional[str] = None
    # Scores can be optionally included if the examples are generated from model
    # predictions. Indexes of `scores` should correspond to indexes of `docs`.
    scores: Optional[typing.Iterable[float]] = None
    # Optional metadata.
    metadata: Optional[ExampleMetadata] = None

# Extract examples from jsonl file
def read_examples(filepath: str) -> List[Example]:
    examples_json = read_jsonl(filepath)
    examples = [Example(**example) for example in examples_json]
    for example in examples:
        example.metadata = ExampleMetadata(**example.metadata)
    return examples

def read_jsonl(file: str):
    """Read jsonl file to a List of Dicts."""
    data = []
    with jsonlines.open(file, mode="r") as jsonl_reader:
        for json_line in jsonl_reader:
            try:
                data.append(json_line)
            except jsonlines.InvalidLineError as e:
                print("Failed to parse line: `%s`" % json_line)
                raise e
    print("Loaded %s lines from %s." % (len(data), file))
    return data


# Read document
@dataclass
class Document:
    """Represents a document with its title and text."""
    # Document title (should be unique in corpus).
    title: str
    # Document text.
    text: str


def read_documents(filepath: str) -> List[Document]:
    documents_json = read_jsonl(filepath)
    return [Document(**document) for document in documents_json]

def read_documents_flatten(filepath: str) -> List[Dict]:
    documents_json = read_jsonl(filepath)
    return [
        dict(
            title=document["title"],
            text=document["text"],
        ) for document in documents_json
    ]


# Write file
def write_jsonl(filepath, data):
    with jsonlines.open(filepath, mode="w") as jsonl_file:
        jsonl_file.write_all(data)
    print(f"Data has been dump to {filepath}")

def write_json(filepath, data):
    with open(filepath, "w") as json_file:
        json.dump(data, json_file)
    print(f"Data has been dump to {filepath}")

@dataclass
class TrainingExample:
    """Represents a training example for a retrieval model."""
    query: str
    positive_docs: List[Document]
    negative_docs: List[Document]
    hard_negative_docs: List[Document]

# def normalize(text: str):
#     text = text.replace("\n", " ")
    