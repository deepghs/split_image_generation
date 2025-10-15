import os
from collections import defaultdict
from typing import List, Dict
from typing import Type

from PIL import Image
from cheesechaser.datapool import IncrementIDDataPool
from cheesechaser.datapool import YandeWebpDataPool, ZerochanWebpDataPool, GelbooruWebpDataPool, \
    KonachanWebpDataPool, AnimePicturesWebpDataPool, DanbooruNewestWebpDataPool, Rule34WebpDataPool
from hfutils.utils import TemporaryDirectory


def quick_webp_pool(site_name: str, level: int = 3) -> Type[IncrementIDDataPool]:
    repo_id = f'deepghs/{site_name}-webp-4Mpixel'

    class _QuickWebpDataPool(IncrementIDDataPool):
        def __init__(self, revision: str = 'main'):
            IncrementIDDataPool.__init__(
                self,
                data_repo_id=repo_id,
                data_revision=revision,
                idx_repo_id=repo_id,
                idx_revision=revision,
                base_level=level,
            )

    return _QuickWebpDataPool


_SITE_CLS = {
    'danbooru': DanbooruNewestWebpDataPool,
    'yandere': YandeWebpDataPool,
    'zerochan': ZerochanWebpDataPool,
    'gelbooru': GelbooruWebpDataPool,
    'konachan': KonachanWebpDataPool,
    'anime_pictures': AnimePicturesWebpDataPool,
    'rule34': Rule34WebpDataPool,
    'nozomi': 'nozomi_standalone',
}


def _get_from_ids(site_name: str, ids: List[int]) -> Dict[int, Image.Image]:
    with TemporaryDirectory() as td:
        site_cls = _SITE_CLS.get(site_name) or quick_webp_pool(site_name, 3)
        if isinstance(site_cls, str):
            site_cls = quick_webp_pool(site_cls, 3)
        datapool = site_cls()
        datapool.batch_download_to_directory(
            resource_ids=ids,
            dst_dir=td,
        )

        retval = {}
        for file in os.listdir(td):
            id_ = int(os.path.splitext(file)[0])
            image = Image.open(os.path.join(td, file))
            image.load()
            retval[id_] = image

        return retval


def _get_from_raw_ids(ids: List[str]) -> Dict[str, Image.Image]:
    _sites = defaultdict(list)
    for id_ in ids:
        site_name, num_id = id_.rsplit('_', maxsplit=1)
        num_id = int(num_id)
        _sites[site_name].append(num_id)

    _retval = {}
    for site_name, site_ids in _sites.items():
        _retval.update({
            f'{site_name}_{id_}': image
            for id_, image in _get_from_ids(site_name, site_ids).items()
        })
    return _retval


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    raw = _get_from_raw_ids(['danbooru_699998'])
    print(raw)

    plt.imshow(raw['danbooru_699998'])
    plt.show()

    _ = _get_from_raw_ids([
        'danbooru_6999999', 'danbooru_6999998', 'danbooru_6999997', 'danbooru_6999996', 'danbooru_6999995',
        'danbooru_6999994', 'danbooru_6999993', 'danbooru_6999992', 'danbooru_6999991', 'danbooru_6999990',
        'danbooru_6999989', 'danbooru_6999988', 'danbooru_6999987', 'danbooru_6999986', 'danbooru_6999985',
        'danbooru_6999984', 'danbooru_6999983', 'danbooru_6999982', 'danbooru_6999981', 'danbooru_6999980',
        'danbooru_6999979', 'danbooru_6999978', 'danbooru_6999977', 'danbooru_6999976', 'danbooru_6999975',
        'danbooru_6999974', 'danbooru_6999973', 'danbooru_6999972', 'danbooru_6999971', 'danbooru_6999970',
        'danbooru_6999969', 'danbooru_6999968', 'danbooru_6999967', 'danbooru_6999966', 'danbooru_6999965',
        'danbooru_6999964', 'danbooru_6999963', 'danbooru_6999962', 'danbooru_6999961', 'danbooru_6999960',
        'danbooru_6999959', 'danbooru_6999958', 'danbooru_6999957', 'danbooru_6999956', 'danbooru_6999955',
        'danbooru_6999954', 'danbooru_6999953', 'danbooru_6999952', 'danbooru_6999951', 'danbooru_6999950',
        'danbooru_6999949', 'danbooru_6999948', 'danbooru_6999947', 'danbooru_6999946', 'danbooru_6999945',
        'danbooru_6999944', 'danbooru_6999943', 'danbooru_6999942', 'danbooru_6999941', 'danbooru_6999940',
        'danbooru_6999939', 'danbooru_6999938', 'danbooru_6999937', 'danbooru_6999936', 'danbooru_6999935',
        'danbooru_6999934', 'danbooru_6999933', 'danbooru_6999932', 'danbooru_6999931', 'danbooru_6999930',
        'danbooru_6999929', 'danbooru_6999928', 'danbooru_6999927', 'danbooru_6999926', 'danbooru_6999925',
        'danbooru_6999924', 'danbooru_6999923', 'danbooru_6999922', 'danbooru_6999921', 'danbooru_6999920',
        'danbooru_6999919', 'danbooru_6999918', 'danbooru_6999917', 'danbooru_6999916', 'danbooru_6999915',
        'danbooru_6999914', 'danbooru_6999913', 'danbooru_6999912', 'danbooru_6999911', 'danbooru_6999910',
        'danbooru_6999909', 'danbooru_6999908', 'danbooru_6999907', 'danbooru_6999906', 'danbooru_6999905',
        'danbooru_6999904', 'danbooru_6999903', 'danbooru_6999902', 'danbooru_6999901', 'danbooru_6999900',
    ])
