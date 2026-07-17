# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import copy
import os
from pathlib import Path
from typing import Mapping, Union
from urllib.parse import unquote, urlparse

import torch.hub
from omegaconf import DictConfig, OmegaConf


__configs__ = {
    'ks': {
        'url': 'https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/ks.pth',
        'feature': 'normal',
        'geometry': 'kernel',
        'voxel_size': 0.1,
        'kernel_dim': 4,
        'tree_depth': 4,
        'adaptive_depth': 2,
        'unet': {
            'f_maps': 32
        },
        'udf': {
            'enabled': True
        },
        'interpolator': {
            'n_hidden': 2,
            'hidden_dim': 16
        },
        'solver': {
            'pos_weight': 1e4,
            'normal_weight': 1e4
        },
        'density_range': [1.0, 20.0]
    },
    'snet': {
        'parent': 'ks',
        'url': 'https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/snet-n3k-wnormal.pth',
        'voxel_size': 0.02,
        'kernel_dim': 16,
        'adaptive_depth': 1,
        'udf': {
            'enabled': False
        },
        'interpolator': {
            'n_hidden': 2,
            'hidden_dim': 32
        },
        'density_range': None
    },
    'snet-wonormal': {
        'parent': 'snet',
        'url': 'https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/snet-n3k-wonormal.pth',
        'feature': 'none'
    }
}


def get_hparams(config: Union[str, Mapping]):
    if isinstance(config, str):
        assert config in __configs__.keys(), f"Config '{config}' not found! " \
                                             f"Available ones are {list(__configs__.keys())}"
        hparams = DictConfig(copy.deepcopy(__configs__[config]))
    else:
        hparams = DictConfig(config)

    if "parent" in hparams:
        parent_hparams = get_hparams(hparams.parent)
        hparams = OmegaConf.merge(parent_hparams, hparams)
        del hparams["parent"]

    return hparams


def local_checkpoint_path(url: str):
    checkpoint_dir = os.environ.get("NKSR_CHECKPOINTS_DIR", "").strip()
    if not checkpoint_dir or not url.startswith("http"):
        return None
    filename = Path(unquote(urlparse(url).path)).name
    return Path(checkpoint_dir).expanduser() / filename


def load_checkpoint_from_url(url: str):
    local_path = local_checkpoint_path(url)
    if local_path is not None:
        if not local_path.is_file():
            raise FileNotFoundError(f"NKSR checkpoint not found: {local_path}")
        url = str(local_path)

    if url.startswith("http"):
        return torch.hub.load_state_dict_from_url(url)
    elif url.startswith("gdrive"):
        expand_url = f"https://drive.google.com/uc?export=download&id={url.split('://')[1]}"
        return torch.hub.load_state_dict_from_url(expand_url)
    else:
        state_dict_data = torch.load(url)['state_dict']
        state_dict_data = {
            k.split("network.")[1] if "network." in k else k: v for k, v in state_dict_data.items()
        }
        return {'state_dict': state_dict_data}
