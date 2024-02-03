from dataclasses import dataclass
import re
from typing import Union, List


# QUERY_TEMPLATE = {
#     '_',                            # A
#     '_ or _',                       # A or B
#     '_ that are not _',             # A \ B
#     '_ that are also _',            # A and B 
#     '_ or _ or _',                  # A or B or C
#     '_ that are also _ but not _',  # A and B \ C 
#     '_ that are also both _ and _'  # A and B and C
# }


# @dataclass
# class QuestSample:
#     query: str
#     original_query: str
#     metadata: dict
#     positive_passages: list
#     negative_passages: list

# def normalize_query(sample):
#     """
#     Query segmentation and normalization
#     e.g., "Philippine remakes of South Korean films or 2010s prison dramas" 
#     -> "Philippine remakes of South Korean films" and "or 2010s prison dramas"
#     """
#     query = sample['query']
#     original_query = sample['original_query']
#     template = sample['metadata']['template']

#     query = query.replace("’", "'")
#     extracted_queries = extract_query(original_query, template)
#     return query, extracted_queries
    

# def normalizer_doc(doc: str) -> str:
#     doc = doc.replace("’", "'").replace("'''", "")
#     return doc

# def segment_query(query: str, template: str):
#     template_mapping = {
#         '_': (['_'], ['']),
#         '_ or _': (['or'], ['or ']),
#         '_ that are not _': (['that are not'], ['and not ']),
#         '_ that are also _': (['that are also'], ['and ']),
#         '_ or _ or _': (['or', 'or'], ['or ', 'or ']),
#         '_ that are also _ but not _': (['that are also', 'but not'], ['and ', 'and not ']),
#         '_ that are also both _ and _': (['that are also both', 'and'], ['and ', 'and '])
#     }

#     if template in template_mapping:
#         delimiter, fill = template_mapping[template]
#         return _segment(query, delimiter, fill)
#     else:
#         raise ValueError(f"Invalid template: {template}")
    
def segment_query(query: str, template: str):
    if template == '_':
        return [query]
    elif template == '_ or _':
        # -> ["A", "or B""]
        return _segment(query, 'or', 'or ')
    elif template == '_ that are not _':
        # -> ["A", "and not B"]
        return _segment(query, 'that are not', 'and not ')
    elif template == '_ that are also _':
        # -> ["A", "and B"]
        return _segment(query, 'that are also', 'and ')
    elif template == '_ or _ or _':
        # -> ["A", "or B", "or C"]
        return _segment(query, ['or', 'or'], ['or ', 'or '])
    elif template == '_ that are also _ but not _':
        # -> ["A", "and B", "and not C"]
        return _segment(query, ['that are also', 'but not'], ['and ', 'and not '])
    elif template == '_ that are also both _ and _':
        # -> ["A", "and B", "and C"]
        return _segment(query, ['that are also both', 'and'], ['and ', 'and '])
    else:
        raise ValueError(f"Invalid template: {template}")

def _segment(
        query: str, 
        delimiter: Union[str, List[str]], 
        fill: Union[str, List[str]]
    ) -> List[str]:

    if isinstance(fill, str):
        query = query.split(delimiter)
        query = [query[0], fill + query[1]]
    elif isinstance(fill, list):
        pattern = r'\s+|\s+'.join(map(re.escape, delimiter))
        query = re.split(pattern, query)
        query = [q.strip() for q in query if q.strip()]
        assert len(query) == 1 + len(fill)
        query = [query[0], fill[0] + query[1], fill[1] + query[2]]
    else:
        raise ValueError("Invalid fill type. Expected str or list of str.")
    
    return query
    
def extract_query(query: str, template: str):
    pattern = r'<mark>(.*?)</mark>'
    qlist = re.findall(pattern, query)
    if template == '_':
        return qlist# + ["", ""]
    elif template == '_ or _':
        # -> ["A", "or B""]
        return [qlist[0], "or " + qlist[1]] #, ""]
    elif template == '_ that are not _':
        # -> ["A", "and not B"]
        return [qlist[0], "and not " + qlist[1]] #, ""]
    elif template == '_ that are also _':
        # -> ["A", "and B"]
        return [qlist[0], "and " + qlist[1]] #, ""]
    elif template == '_ or _ or _':
        # -> ["A", "or B", "or C"]
        return [qlist[0], "or " + qlist[1], "or " + qlist[2]]
    elif template == '_ that are also _ but not _':
        # -> ["A", "and B", "and not C"]
        return [qlist[0], "and " + qlist[1], "and not " + qlist[2]]
    elif template == '_ that are also both _ and _':
        # -> ["A", "and B", "and C"]
        return [qlist[0], "and " + qlist[1], "and " + qlist[2]]
    else:
        raise ValueError(f"Invalid template: {template}")
    
def extract_query_for_embs(original_query: str):
    pattern = r'<mark>(.*?)</mark>'
    qlist = re.findall(pattern, original_query)
    return qlist
    
def extract_query_or(query: str, original_query: str, template: str):
    pattern = r'<mark>(.*?)</mark>'
    qlist = re.findall(pattern, original_query)
    if template == '_':
        return qlist# + ["", ""]
    elif template == '_ or _':
        # -> ["A", "or B""]
        return [query]
    elif template == '_ that are not _':
        # -> ["A", "and not B"]
        return [qlist[0], "not " + qlist[1]] #, ""]
    elif template == '_ that are also _':
        # -> ["A", "and B"]
        return [qlist[0], qlist[1]] #, ""]
    elif template == '_ or _ or _':
        # -> ["A", "or B", "or C"]
        return [query]
    elif template == '_ that are also _ but not _':
        # -> ["A", "and B", "and not C"]
        return [qlist[0], qlist[1], "not " + qlist[2]]
    elif template == '_ that are also both _ and _':
        # -> ["A", "and B", "and C"]
        return [qlist[0], qlist[1], qlist[2]]
    else:
        raise ValueError(f"Invalid template: {template}")


def set_seed(seed: int = 19980406):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_yaml_file(file_path):
    import yaml  
    with open(file_path, "r") as file:  
        config = yaml.safe_load(file)  
    return config

def normalize_document(document: str):
    document = document.replace("\n", " ").replace("’", "'").replace("'''", "")
    if document.startswith('"'):
        document = document[1:]
    if document.endswith('"'):
        document = document[:-1]
    return document

def normalize_query(question: str) -> str:
    question = question.replace("’", "'")
    return question


def get_linear_scheduler(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):
    from torch.optim.lr_scheduler import LambdaLR
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)