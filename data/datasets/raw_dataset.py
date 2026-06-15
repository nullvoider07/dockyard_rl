from datasets import Dataset
from dockyard_rl.data import PreferenceDatasetConfig, ResponseDatasetConfig
from dockyard_rl.data.interfaces import (
    TaskDataPreProcessFnCallable,
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from dockyard_rl.data.processors import PROCESSOR_REGISTRY

class RawDataset:
    # change to ResponseDatasetConfig | PreferenceDatasetConfig once preference dataset is refactored
    data_config: ResponseDatasetConfig | PreferenceDatasetConfig
    dataset: Dataset
    # `val_dataset` is used only when current dataset is used for both training and validation
    val_dataset: Dataset | None
    processor: TaskDataProcessFnCallable
    task_spec: TaskDataSpec
    preprocessor: TaskDataPreProcessFnCallable | None = None

    def split_train_validation(self, test_size: float, seed: int):
        if test_size > 0:
            split_dataset = self.dataset.train_test_split(
                test_size=test_size, seed=seed
            )
            self.dataset = split_dataset["train"]
            self.val_dataset = split_dataset["test"]

    def set_processor(self):
        processor_name = "default"
        if "processor" in self.data_config:
            processor_name = self.data_config[
                "processor"  # pyrefly: ignore[typed-dict-key-error]  `processor` is only required for response datasets and will be removed after data processor is refactored
            ]
        assert processor_name in PROCESSOR_REGISTRY, (
            f"Processor {processor_name} not found in PROCESSOR_REGISTRY. "
            "Please call dockyard_rl.data.processors.register_processor() to register the processor."
        )
        self.processor = PROCESSOR_REGISTRY[processor_name]

    def set_task_spec(
        self, data_config: ResponseDatasetConfig | PreferenceDatasetConfig
    ):
        self.data_config = data_config
        system_prompt_file = self.data_config.get("system_prompt_file", None)
        prompt_file = self.data_config.get("prompt_file", None)
        self.task_spec = TaskDataSpec(
            task_name=self.task_name,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )