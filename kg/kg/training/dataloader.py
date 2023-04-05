# ===============================================
#
# Custom Dataloader
#
# ===============================================
from torch.utils.data import DataLoader


class InfiniteDataLoader(DataLoader):
    """
    Infinitely Looping dataloader

    From: https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
