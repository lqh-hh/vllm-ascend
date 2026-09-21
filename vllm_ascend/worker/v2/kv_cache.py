# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from contextlib import contextmanager
from functools import wraps

import torch
from vllm.v1.worker.gpu import model_runner


def _cache_tensors(cache):
    if isinstance(cache, torch.Tensor):
        yield cache
    elif isinstance(cache, (tuple, list)):
        for component in cache:
            yield from _cache_tensors(component)
    elif cache is not None:
        raise TypeError(f"Unsupported KV cache component: {type(cache).__name__}")


@contextmanager
def kv_cache_init_wrapper():
    """Separate the runner's tensor inventory from Ascend's per-layer caches.

    vLLM builds its block-copy inventory from init_kv_cache().values()
    and requires Tensor values. Ascend attention/MLA/Mamba and KV connectors
    still consume per-layer tuples/lists. Let upstream allocate and bind those
    original views, expose their tensors for inventory construction only, and
    restore the original mapping at the connector boundary. No tensors are
    copied or reallocated. Like graph_manager_wrapper, this is scoped to the
    worker's synchronous initialization.
    """
    original_init = model_runner.init_kv_cache
    original_connector = model_runner.get_kv_connector
    layer_caches = None
    tensor_caches = None

    @wraps(original_init)
    def init_kv_cache(*args, **kwargs):
        nonlocal layer_caches, tensor_caches
        layer_caches = original_init(*args, **kwargs)
        tensor_caches = {
            (name, index): tensor
            for name, cache in layer_caches.items()
            for index, tensor in enumerate(_cache_tensors(cache))
        }
        return tensor_caches

    @wraps(original_connector)
    def get_kv_connector(vllm_config, kv_caches_dict):
        if tensor_caches is not None and kv_caches_dict is tensor_caches:
            kv_caches_dict = layer_caches
        return original_connector(vllm_config, kv_caches_dict)

    try:
        model_runner.init_kv_cache = init_kv_cache
        model_runner.get_kv_connector = get_kv_connector
        yield
    finally:
        model_runner.init_kv_cache = original_init
        model_runner.get_kv_connector = original_connector
