import logging
from omegaconf import DictConfig
from data.biencoder_data import JsonlQADataset
import os
import glob
import hydra

logger = logging.getLogger(__name__)


class BiencoderDatasetsCfg(object):
    def __init__(self, cfg: DictConfig):
        ds_cfg = cfg.datasets
        self.train_datasets_names = cfg.train_datasets
        logger.info("train_datasets: %s", self.train_datasets_names)
        self.train_datasets = _init_datasets(self.train_datasets_names, ds_cfg)
        self.dev_datasets_names = cfg.dev_datasets
        logger.info("dev_datasets: %", self.dev_datasets_names)
        self.dev_datasets = _init_datasets(self.dev_datasets_names, ds_cfg)
        self.sampling_rates = cfg.train_sampling_rates

def _init_datasets(datasets_names, ds_cfg: DictConfig):
    if isinstance(datasets_names, str):
        return [_init_dataset(datasets_names, ds_cfg)]
    elif datasets_names:
        return [_init_dataset(name, ds_cfg) for name in datasets_names]
    else:
        return []

def _init_dataset(name: str, ds_cfg: DictConfig):
    if os.path.exists(name):
        return JsonlQADataset(name)
    elif glob.glob(name):
        files = glob.glob(name)
        return [_init_dataset(f, ds_cfg) for f in files]
    if name not in ds_cfg:
        raise RuntimeError("Cannot find dataset location/config for: {}".format(name))
    return hydra.utils.instantiate(ds_cfg[name])