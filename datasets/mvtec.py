from .base import BaseDataset


MVTEC_CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]


class MVTecDataset(BaseDataset):
    def __init__(self, args, is_train=True, class_name=None):
        if class_name is not None:
            class_list = [class_name]
        else:
            class_list = MVTEC_CLASS_NAMES
        super().__init__(args, is_train, class_list)
