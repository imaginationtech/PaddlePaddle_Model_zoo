from .operators import DecodeImage
from .operators import ResizeImage
from .operators import CropImage
from .operators import NormalizeImage
from .operators import ToCHWImage


def transform(data, ops=[]):
    """ transform """
    for op in ops:
        data = op(data)
    return data
