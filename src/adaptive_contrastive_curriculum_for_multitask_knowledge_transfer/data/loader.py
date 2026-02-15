"""MMLU dataset loader."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class MMLUDataset(TorchDataset):
    """PyTorch Dataset for MMLU."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """Initialize MMLU dataset.

        Args:
            data: List of data samples
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with tokenized inputs and labels
        """
        sample = self.data[idx]

        # Format question with answer choices
        question = sample['question']
        choices = sample['choices']
        formatted_text = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            formatted_text += f"{chr(65+i)}. {choice}\n"

        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(sample['answer'], dtype=torch.long),
            'task_ids': torch.tensor(sample['task_id'], dtype=torch.long),
            'subject': sample['subject'],
        }


class MMLUDataLoader:
    """Data loader for MMLU benchmark."""

    # MMLU task categories
    STEM_TASKS = [
        'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_physics',
        'computer_security', 'conceptual_physics', 'electrical_engineering',
        'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_mathematics', 'high_school_physics',
        'high_school_statistics', 'machine_learning',
    ]

    HUMANITIES_TASKS = [
        'formal_logic', 'high_school_european_history', 'high_school_us_history',
        'high_school_world_history', 'international_law', 'jurisprudence',
        'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
        'prehistory', 'professional_law', 'world_religions',
    ]

    SOCIAL_SCIENCES_TASKS = [
        'econometrics', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_microeconomics',
        'high_school_psychology', 'human_sexuality', 'professional_psychology',
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
    ]

    OTHER_TASKS = [
        'anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine',
        'global_facts', 'human_aging', 'management', 'marketing', 'medical_genetics',
        'miscellaneous', 'nutrition', 'professional_accounting', 'professional_medicine',
        'virology',
    ]

    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        """Initialize data loader.

        Args:
            config: Data configuration dictionary
            tokenizer: Hugging Face tokenizer
        """
        self.config = config
        self.tokenizer = tokenizer
        self.task_to_id: Dict[str, int] = {}
        self.id_to_task: Dict[int, str] = {}

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load and split MMLU dataset.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("Loading MMLU dataset...")

        all_tasks = (
            self.STEM_TASKS + self.HUMANITIES_TASKS +
            self.SOCIAL_SCIENCES_TASKS + self.OTHER_TASKS
        )

        # Create task ID mapping
        for i, task in enumerate(all_tasks):
            self.task_to_id[task] = i
            self.id_to_task[i] = task

        train_data = []
        val_data = []
        test_data = []

        for task in all_tasks:
            try:
                # Load individual task
                dataset = load_dataset('cais/mmlu', task)

                # Extract samples
                task_samples = self._process_task_samples(
                    dataset, task, self.task_to_id[task]
                )

                # Split data
                max_samples = self.config.get('max_samples_per_task')
                if max_samples:
                    task_samples = task_samples[:max_samples]

                n_samples = len(task_samples)
                train_end = int(n_samples * self.config.get('train_split', 0.8))
                val_end = train_end + int(n_samples * self.config.get('val_split', 0.1))

                train_data.extend(task_samples[:train_end])
                val_data.extend(task_samples[train_end:val_end])
                test_data.extend(task_samples[val_end:])

            except Exception as e:
                logger.warning(f"Failed to load task {task}: {e}")
                continue

        logger.info(
            f"Loaded {len(train_data)} train, {len(val_data)} val, "
            f"{len(test_data)} test samples from {len(all_tasks)} tasks"
        )

        return train_data, val_data, test_data

    def _process_task_samples(
        self, dataset: DatasetDict, task_name: str, task_id: int
    ) -> List[Dict[str, Any]]:
        """Process samples from a single task.

        Args:
            dataset: Dataset dictionary
            task_name: Name of the task
            task_id: Integer ID for the task

        Returns:
            List of processed samples
        """
        samples = []

        # Combine all splits
        all_splits = []
        for split in ['test', 'validation', 'dev']:
            if split in dataset:
                all_splits.append(dataset[split])

        for split_data in all_splits:
            for item in split_data:
                samples.append({
                    'question': item['question'],
                    'choices': item['choices'],
                    'answer': item['answer'],
                    'subject': task_name,
                    'task_id': task_id,
                })

        return samples

    def create_dataloaders(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        batch_size: int,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            batch_size: Batch size
            num_workers: Number of data loading workers

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        max_length = self.config.get('max_seq_length', 512)

        train_dataset = MMLUDataset(train_data, self.tokenizer, max_length)
        val_dataset = MMLUDataset(val_data, self.tokenizer, max_length)
        test_dataset = MMLUDataset(test_data, self.tokenizer, max_length)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    def get_task_category(self, task_name: str) -> str:
        """Get category for a task.

        Args:
            task_name: Name of the task

        Returns:
            Category name (STEM, Humanities, Social Sciences, Other)
        """
        if task_name in self.STEM_TASKS:
            return "STEM"
        elif task_name in self.HUMANITIES_TASKS:
            return "Humanities"
        elif task_name in self.SOCIAL_SCIENCES_TASKS:
            return "Social Sciences"
        else:
            return "Other"
