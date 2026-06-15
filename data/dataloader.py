from torchdata.stateful_dataloader import StatefulDataLoader

class MultipleDataloaderWrapper:
    """Wrapper for multiple dataloaders.

    This wrapper is used to sample data from multiple dataloaders using a custom
    dataloader function.

    When a single dataloader is exhausted, the data iterator must be reset in the
    custom dataloader function (as demonstrated in
    `examples/configs/custom_dataloader/custom_dataloader.py`).
    This design ensures that the MultipleDataloaderWrapper operates as an infinite
    iterator, where __next__() will not raise StopIteration and __len__() is not
    supported.
    """

    def __init__(
        self,
        expected_num_prompts: int,
        data_config: dict,
        dataloaders: dict[str, StatefulDataLoader],
    ):
        self.expected_num_prompts = expected_num_prompts
        self.data_config = data_config
        self.dataloaders = dataloaders

        # init data iterators
        self.data_iterators = {
            task_name: iter(dataloader)
            for task_name, dataloader in dataloaders.items()
        }

        # custom dataloader function to decide how to sample the data from the dataloaders
        self.custom_dataloader_func = self._load_custom_dataloader_func()
        # records to pass additional information to the custom dataloader function
        self.records = {}

    def _load_custom_dataloader_func(self):
        import sys
        from pathlib import Path

        from hydra.utils import get_method

        project_root_path = Path(__file__).absolute().parents[2]
        sys.path = [str(project_root_path)] + sys.path

        return get_method(self.data_config["custom_dataloader"])

    def __iter__(self):
        return self

    def __next__(self):
        # sample data from the dataloaders
        result, self.data_iterators = self.custom_dataloader_func(
            self.data_iterators, self.dataloaders, **self.records
        )

        # check if the number of prompts is expected
        assert len(result["message_log"]) == self.expected_num_prompts, (
            f"Expected {self.expected_num_prompts} prompts, but got {len(result['message_log'])}"
        )

        # reset records
        self.records = {}

        return result

    def set_records(self, records: dict):
        """Set the records for the custom dataloader.

        Records are used to pass additional information to the custom dataloader
        function to decide how to sample the data from the dataloaders.
        """
        self.records.update(records)