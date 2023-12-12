from dataclasses import dataclass
import logging
from torch import Tensor as T
import torch
from omegaconf import DictConfig
import hydra
import math
import random
from typing import List, Iterator, Callable, Tuple, Optional
import itertools


logger = logging.getLogger()

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
    relevance_ratings: Optional[typing.Dict[str,
                                                    typing.Sequence[str]]] = None
    evidence_ratings: Optional[typing.Dict[str,
                                                    typing.Sequence[str]]] = None
    # The nested value is a map from query substring to document substring.
    attributions: Optional[typing.Dict[str, typing.Sequence[typing.Dict[
      str, str]]]] = None

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


# Read document
@dataclass
class Document:
    """Represents a document with its title and text."""
    # Document title (should be unique in corpus).
    title: str
    # Document text.
    text: str


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Title will be put before context
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        raise NotImplementedError

    def get_separator_id(self) -> T:
        raise NotImplementedError
    
    def get_pad_id(self) -> T:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T) -> T:
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError
    
    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError
    
    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError
    
    def get_token_id(self, token: str) -> int:
        raise NotImplementedError


def normalize_query(query: str):
    query = query.replace("’", "'")
    return query


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tensorizer: Tensorizer):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position
    
    def get_positions(self, input_ids: T, tensorizer: Tensorizer):
        return self.static_position
    

DEFAULT_SELECTOR = RepStaticPosTokenSelector()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        selector: DictConfig,
        special_token: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        encoder_type: str = None,
    ):
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix
        self.data = []

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        raise NotImplementedError
    
    def calc_total_data_len(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def _process_query(self, query: str):
        query = normalize_query(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix
        return query
    

class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """

    def __init__(
        self,
        dataset: Dataset,
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 1,
        shuffle: bool = True,
        shuffle_seed: int = 0,
        offset: int = 1,
        strict_batch_size: bool = False,
    ):
        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shard = num_shards
        self.iteration = offset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size
        self.shard_start_idx = -1
        self.shard_end_idx = -1
        self.max_iterations = 0

    def calculate_shards(self):
        """ 
        To calculate shard positions
        """
        logger.info("Calculating shard positions")
        shards_num = max(self.num_shard, 1)
        shard_id = max(self.shard_id, 0)

        total_size = self.dataset.calc_total_data_len()
        samples_per_shard = math.ceil(total_size / shards_num)

        self.shard_start_idx = shard_id * samples_per_shard
        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)
        
        self.max_iterations = (
            math.ceil(
                samples_per_shard / self.batch_size
            ) if self.strict_batch_size else samples_per_shard // self.batch_size
        )

        logger.info(
            "samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d",
            samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations,
        )

    def load_data(self):
        self.calculate_shards()
        self.dataset.load_data()
        logger.info("Sharded dataset data %d", len(self.dataset))

    def total_data_len(self) -> int:
        return len(self.dataset)
    
    def iterations_num(self) -> int:
        return self.max_iterations - self.iteration
    
    def max_iterations_num(self) -> int:
        return self.max_iterations
    
    def get_iteration(self) -> int:
        return self.iteration
    
    def apply(self, visitor_func: Callable):
        for sample in self.dataset:
            visitor_func(sample)
    
    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shared_indices = indices[self.shard_start_idx: self.shard_end_idx]
        return shared_indices

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
        max_iterations = self.max_iterations - self.iteration
        shard_indices = self.get_shard_indices(epoch)

        for i in range(self.iteration * self.batch_size, len(shard_indices), self.batch_size):
            items_idxs = shard_indices[i: i + self.batch_size]
            if self.strict_batch_size and len(items_idxs) < self.batch_size:
                logger.debug("Extending batch to max size")
                items_idxs.extend(shard_indices[: self.batch_size - len(items)])
            self.iteration += 1
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        while self.iteration < max_iterations:
            logger.debug("Fulfilling non-complete shard={}".format(self.shard_id))
            self.iteration += 1
            items_idxs = shard_indices[: self.batch_size]
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        logger.info("Finished iterating, iteration={}, shard={}".format(self.iteration, self.shard_id))
        # reset iteration
        self.iteration = 0
    
    def iterate_ds_sampled_data(self, num_iteration: int, epoch: int = 0) -> Iterator[List]:
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch)
        cycle_it = itertools.cycle(shard_indices)
        for _ in range(num_iteration):
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1
            items = [self.dataset[idx] for idx in items_idxs]
            yield items
        
        logger.info("Finished iterating, iteration={}, shard={}".format(self.iteration, self.shard_id))
        self.iteration = 0
    
    def get_dataset(self) -> Dataset:
        return self.dataset
    

class LocalShardedDataIterator(ShardedDataIterator):
    def load_data(self):
        self.calculate_shards()
        self.dataset.load_data(start_pos=self.shard_start_idx, end_pos=self.shard_end_idx)
        logger.info("Sharded dataset data %d", len(self.dataset))
    
    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices
        return shard_indices
    

class MultiSetDataIterator(object):
    """
    Iterator over multiple data sources. 
    Useful when all samples form a single batch should be from the same dataset.
    """

    def __init__(
        self,
        datasets: List[ShardedDataIterator],
        shuffle_seed: int = 0,
        shuffle: bool = True,
        sampling_rate: List = [],
        rank: int = 0,
    ):
        # randomize data loading
        ds_list_copy = [ds for ds in datasets]
        rnd = random.Random(rank)
        rnd.shuffle(ds_list_copy)
        [ds.load_data() for ds in ds_list_copy]

        self.iterables = datasets
        data_lengths = [it.total_data_len() for it in datasets]
        self.total_data = sum(data_lengths)
        logger.info("rank=%d; Multi set data size %s", rank, data_lengths)
        logger.info("rank=%d; Multi set total data %s", rank, self.total_data)
        logger.info("rank=%d; Multi set sampling_rates %s", rank, sampling_rate)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0
        self.rank = rank

        if sampling_rate:
            self.max_its_pr_ds = [int(ds.max_iterations_num() * sampling_rate[i]) for i, ds in enumerate(datasets)]
        else:
            self.max_its_pr_ds = [ds.max_iterations_num() for ds in datasets]
        
        self.max_iterations = sum(self.max_its_pr_ds)
        logger.info("rank=%d; Multi set max_iterations per dataset %s", rank, self.max_its_pr_ds)
        logger.info("rank=%d; Multi set max_iterations %d", rank, self.max_iterations)

    def total_data_len(self) -> int:
        return self.total_data

    def get_max_iterations(self):
        return self.max_iterations

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:
        logger.info("rank=%d; Iteration starts", self.rank)
        logger.info(
            "rank=%d; Multi set iteration: iteration ptr per set %s",
            self.rank,
            [it.get_iteration() for it in self.iterables]
        )
        
        data_src_indices = []
        iterators = []
        for source, src_its in enumerate(self.max_its_pr_ds):
            logger.info(
                "rank=%d; Multi set iteration: source %d, batches to be taken: %s",
                self.rank,
                source,
                src_its,
            )
            data_src_indices.extend([source] * src_its)

            iterators.append(
                self.iterables[source].iterate_ds_sampled_data(src_its, epoch=epoch)
            )
        
        if self.shuffle:
            epoch_rnd = random.Random(self.shuffle + epoch)
            epoch_rnd.shuffle(data_src_indices)
        
        logger.info("rank=%d; data_src_indices len=%d", self.rank, len(data_src_indices))
        for source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)
            if next_item is not None:
                self.iteration += 1
                yield (next_item, source_idx)
            else:
                logger.warning(
                    "rank=%d; Next item in the source %s is None",
                    self.rank,
                    source_idx
                )
        
        logger.info("rank=%d; last iteration %d", self.rank, self.iteration)

        logger.infor(
            "rank=%d; Multi set iteration finished: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables]
        )

        [next(it, None) for it in iterators]

        for it in self.iterables:
            it.iteration = 0
        logger.info(
            "rank=%d; Multi set iteration finished after next: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables]
        )

        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration
    
    def get_dataset(self, ds_id: int) -> Dataset:
        return self.iterables[ds_id].get_dataset()
    
    def get_datasets(self) -> List[Dataset]:
        return [it.get_dataset() for it in self.iterables]