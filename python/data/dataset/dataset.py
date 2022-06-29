from __future__ import print_function


__all__ = [
    "Dataset", "IterableDataset"]


class Dataset(object):

    def __init__(self):
        pass

    def __getitem__(self, idx):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__len__', self.__class__.__name__))


class IterableDataset(Dataset):

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError("'{}' not implement in class "\
                "{}".format('__iter__', self.__class__.__name__))

    def __getitem__(self, idx):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__getitem__', self.__class__.__name__))

    def __len__(self):
        raise RuntimeError("'{}' should not be called for IterableDataset" \
                "{}".format('__len__', self.__class__.__name__))
