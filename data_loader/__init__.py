from .cityscapes import CitySegmentation
from .wire_load import WireSegmentation

datasets = {
    'citys': CitySegmentation,
    'wire': WireSegmentation
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
