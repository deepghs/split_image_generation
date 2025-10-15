from functools import lru_cache
from pprint import pprint

import numpy as np
from huggingface_hub import hf_hub_download

from .site import get_images_from_site

_DEFAULT_MODEL_NAME = 'SwinV2_v3_danbooru_8005009_4GB'


@lru_cache()
def _get_image_ids(model_name: str = _DEFAULT_MODEL_NAME):
    return np.load(hf_hub_download(
        repo_id='deepghs/index_experiments',
        repo_type='model',
        filename=f'{model_name}/ids.npy'
    ))


def get_random_images(count: int = 10, model_name: str = _DEFAULT_MODEL_NAME, max_workers: int = 16):
    image_ids = np.random.choice(_get_image_ids(model_name), count)
    return get_images_from_site(image_ids, max_workers=max_workers)


if __name__ == '__main__':
    pprint(get_random_images())
