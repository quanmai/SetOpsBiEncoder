from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class Arguments(TrainingArguments):
    pass