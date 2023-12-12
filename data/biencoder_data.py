import logging
import os
import glob
import jsonlines
from dataclasses import dataclass
from typing import Dict, List, Tuple

from omegaconf import DictConfig
from utils.data_utils import Dataset




logger = logging.getLogger(__name__)

@dataclass
class BiEncoderPassage:
    text: str
    title: str

@dataclass
class BiEncoderSample:
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]

def get_data_files(source_name) -> List[str]:
    if os.path.exists(source_name) or glob.glob(source_name):
        return glob.glob(source_name)

class JsonlQADataset(Dataset):
    def __init__(
        self,
        file: str,
        selector: DictConfig, 
        special_token: str = None,
        encoder_type: str = None,
        shuffle_positives: bool = False, 
        normalize: bool = False,
        query_special_suffix: str = None, 
        exclude_gold: bool = False,
        total_data_size: int = -1,
    ):
        super().__init__(
            selector,
            special_token=special_token, 
            shuffle_positives=shuffle_positives,
            query_special_suffix=query_special_suffix, 
            encoder_type=encoder_type)
        self.file = file
        self.normalize = normalize
        self.exclude_gold = exclude_gold
        self.total_data_size = total_data_size
        self.data_files = get_data_files(self.file)
        logger.info("Data files: %s", self.data_files)

    def calc_total_data_len(self):
        if self.total_data_size < 0:
            logger.info("Calculating data size")
            for file in self.data_files:
                with jsonlines.open(file, mode="r") as jsonl_reader:
                    for _ in jsonl_reader:
                        self.total_data_size += 1
        logger.info("total_data_size=%d", self.total_data_size)
        return self.total_data_size
    
    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        if self.data:
            return
        
        logger.info("Jsonl loading subset range from %d to %d", start_pos, end_pos)
        if start_pos < 0 and end_pos < 0:
            for file in self.data_files:
                with jsonlines.open(file, mode="r") as jsonl_reader:
                    self.data.extend([l for l in jsonl_reader])
            return
        
        global_sample_id = 0
        for file in self.data_files:
            if global_sample_id >= end_pos:
                break
            with jsonlines.open(file, mode="r") as jsonl_reader:
                for jline in jsonl_reader:
                    if start_pos <= global_sample_id < end_pos:
                        self.data.append(jline)
                    if global_sample_id >= end_pos:
                        break
                    global_sample_id += 1
        logger.info("jsonl loaded data size %d", len(self.data))
        
