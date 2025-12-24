import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from federatedscope.register import register_data
from federatedscope.core.data import ClientData, StandaloneDataDict
from federatedscope.llm.dataloader import get_tokenizer
from federatedscope.llm.dataset.chat_dataset import ChatSFTDataset

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _normalize_client_names(client_cfg) -> Optional[List[str]]:
    if not client_cfg:
        return None
    normalized = []
    for entry in client_cfg:
        if isinstance(entry, str):
            normalized.append(entry)
        elif isinstance(entry, dict):
            if "name" not in entry:
                raise ValueError(f"Client config entries must include 'name': {entry}")
            normalized.append(entry["name"])
        else:
            raise TypeError(f"Unsupported client config entry: {entry}")
    return normalized


def _build_client_data(samples: List[Dict], tokenizer, max_len: int):
    if not samples:
        return None
    return ChatSFTDataset(samples, tokenizer, max_length=max_len)


def load_tulu3_federated_data(config, client_cfgs=None):
    """
    Loader for datasets prepared by ``scripts/prepare_tulu3_federated.py``.
    """
    root = Path(config.data.root)
    cfg = config.data.tulu3_federated
    dataset_root = root / cfg.root
    manifest_path = dataset_root / cfg.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Tulu3 manifest not found at {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    available_clients = {client["name"]: client for client in manifest["clients"]}
    requested_names = _normalize_client_names(cfg.clients)
    if requested_names is None:
        requested_names = list(available_clients.keys())

    missing = [name for name in requested_names if name not in available_clients]
    if missing:
        raise ValueError(f"Clients {missing} not found in manifest {manifest_path}")

    if "@" not in config.model.type:
        raise ValueError("config.model.type must include the tokenizer source, e.g. "
                         "'Llama-2-7b-hf@huggingface_llm'")
    model_name, model_hub = config.model.type.split("@")
    tokenizer, _ = get_tokenizer(model_name, config.data.root,
                                 config.llm.tok_len, model_hub)

    merge_clients = bool(getattr(cfg, "merge_clients", False))
    client_data = {}
    if merge_clients:
        merged_train = []
        merged_val = []
        for name in requested_names:
            entry = available_clients[name]
            train_path = dataset_root / entry["train_file"]
            val_path = dataset_root / entry["val_file"]
            train_samples = _read_jsonl(train_path)
            val_samples = _read_jsonl(val_path)
            if not train_samples:
                raise ValueError(
                    f"Client {name} has no training samples. Expected data at {train_path}")
            merged_train.extend(train_samples)
            merged_val.extend(val_samples)
            logger.info("Client %s -> train=%d, val=%d", name, len(train_samples),
                        len(val_samples))

        train_dataset = _build_client_data(merged_train, tokenizer, config.llm.tok_len)
        val_dataset = _build_client_data(merged_val, tokenizer, config.llm.tok_len)
        client_data[1] = ClientData(config,
                                    train=train_dataset,
                                    val=val_dataset if val_dataset else None,
                                    test=None)
        logger.info("Merged clients -> train=%d, val=%d", len(merged_train),
                    len(merged_val))
    else:
        for idx, name in enumerate(requested_names, start=1):
            entry = available_clients[name]
            train_path = dataset_root / entry["train_file"]
            val_path = dataset_root / entry["val_file"]
            train_samples = _read_jsonl(train_path)
            val_samples = _read_jsonl(val_path)
            train_dataset = _build_client_data(train_samples, tokenizer, config.llm.tok_len)
            val_dataset = _build_client_data(val_samples, tokenizer, config.llm.tok_len)
            if train_dataset is None:
                raise ValueError(
                    f"Client {name} has no training samples. Expected data at {train_path}")
            client_data[idx] = ClientData(config,
                                          train=train_dataset,
                                          val=val_dataset if val_dataset else None,
                                          test=None)
            logger.info("Client %s -> train=%d, val=%d", name, len(train_samples),
                        len(val_samples))

    if len(client_data) == 0:
        raise RuntimeError("No client datasets were loaded. Please verify the manifest and config.")

    if config.federate.client_num != len(client_data):
        config.defrost()
        config.federate.client_num = len(client_data)
        config.freeze()

    data_dict = {0: ClientData(config, train=None, val=None, test=None)}
    data_dict.update(client_data)
    return StandaloneDataDict(data_dict, config), config


def call_tulu3_federated_data(config, client_cfgs):
    if config.data.type.lower() == "tulu3_federated":
        return load_tulu3_federated_data(config, client_cfgs)


register_data("tulu3_federated", call_tulu3_federated_data)
