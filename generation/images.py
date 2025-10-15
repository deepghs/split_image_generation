from functools import lru_cache
from pprint import pprint

import numpy as np
from huggingface_hub import hf_hub_download

from .pool import _get_from_raw_ids

_DEFAULT_MODEL_NAME = 'SwinV2_v3_danbooru_8005009_4GB'


@lru_cache()
def _get_image_ids(model_name: str = _DEFAULT_MODEL_NAME):
    return np.load(hf_hub_download(
        repo_id='deepghs/index_experiments',
        repo_type='model',
        filename=f'{model_name}/ids.npy'
    ))


def _get_random_images(count: int = 10, model_name: str = _DEFAULT_MODEL_NAME):
    image_ids = np.random.choice(_get_image_ids(model_name), count)
    return _get_from_raw_ids(image_ids)


if __name__ == '__main__':
    pprint(_get_random_images())
