import os
import random
import uuid

import yaml
from hfutils.index import tar_cache_reset, hf_tar_cache_reset

from generation.gene.random import random_one
from generation.utils import parallel_call

splits = ['train', 'test', 'valid']
rts = [0.7, 0.1, 0.2]

if __name__ == '__main__':
    hf_tar_cache_reset(10000)
    tar_cache_reset(10000)
    dst_dir = os.environ.get('DST_DIR', '/nfs/box_v0')

    meta_file = os.path.join(dst_dir, 'data.yaml')
    os.makedirs(os.path.dirname(meta_file), exist_ok=True)
    with open(meta_file, 'w') as f:
        yaml.dump({
            'train': '../train/images',
            'val': '../valid/images',
            'test': '../test/images',

            'nc': 1,
            'names': ['box'],
        }, f)


    def _fn(_):
        id_ = uuid.uuid4().hex
        split_name = random.choices(splits, weights=rts, k=1)[0]
        image, usable_bboxes = random_one()

        dst_image_file = os.path.join(dst_dir, split_name, 'images', f'{id_}.jpg')
        os.makedirs(os.path.dirname(dst_image_file), exist_ok=True)
        image.save(dst_image_file)

        dst_labels_file = os.path.join(dst_dir, split_name, 'labels', f'{id_}.txt')
        os.makedirs(os.path.dirname(dst_labels_file), exist_ok=True)
        with open(dst_labels_file, 'w') as f:
            for (x0, y0, x1, y1), _, _ in usable_bboxes:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                width, height = abs(x1 - x0), abs(y1 - y0)
                r_cx, r_cy = cx / image.width, cy / image.height
                r_width, r_height = width / image.width, height / image.height
                print(f'0 {r_cx} {r_cy} {r_width} {r_height}', file=f)


    parallel_call(
        range(1000000),
        fn=_fn,
        desc='Making Dataset',
        max_workers=64,
        max_pending=320,
    )
