from .base import BaseDataset


VISA_CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum',
    'fryum', 'macaroni1', 'macaroni2', 'pcb1',
    'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]


class VisADataset(BaseDataset):
    def __init__(self, args, is_train=True, class_name=None):
        if class_name is not None:
            class_list = [class_name]
        else:
            class_list = VISA_CLASS_NAMES
        super().__init__(args, is_train, class_list)
