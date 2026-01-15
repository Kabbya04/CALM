# File: dataset_manager.py
# Dataset manager for CIFAR-10, CIFAR-100, FashionMNIST, and STL-10

import torch
from torchvision.datasets import CIFAR100, CIFAR10, FashionMNIST, STL10
from torchvision import transforms
from torch.utils.data import Subset

DATASET_CONFIGS = {
    'cifar100': {
        'default_tasks': 10  # 10 tasks of 10 classes
    },
    'cifar10': {
        'default_tasks': 5   # 5 tasks of 2 classes
    },
    'fashionmnist': {
        'default_tasks': 5   # 5 tasks of 2 classes
    },
    'stl10': {
        'default_tasks': 5   # 5 tasks of 2 classes
    }
}


class DatasetManager:
    """
    A unified interface to handle CIFAR-10, CIFAR-100, FashionMNIST, and STL-10 datasets for continual learning experiments.
    """
    def __init__(self, dataset_name, num_tasks=None, k_shot=5):
        """
        Initializes the manager and loads the specified dataset.
        If num_tasks is not provided, it will use the sensible default for that dataset.
        """
        self.dataset_name = dataset_name.lower()
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported. Supported: {list(DATASET_CONFIGS.keys())}")
        
        if num_tasks is None:
            self.num_tasks = DATASET_CONFIGS[self.dataset_name]['default_tasks']
            print(f"No --num_tasks specified. Using smart default of {self.num_tasks} for {self.dataset_name}.")
        else:
            self.num_tasks = num_tasks
            print(f"User override: Using {self.num_tasks} tasks.")
        
        self.k_shot = k_shot
        
        self.train_dataset = None
        self.test_dataset = None
        self.classes_per_task = None
        self.class_names = None
        
        self._load_dataset()
        print(f"DatasetManager initialized for '{self.dataset_name}' with {self.num_tasks} tasks.")

    def _load_dataset(self):
        """Internal method to load the correct dataset based on the name."""
        
        if self.dataset_name == 'cifar100':
            transform = transforms.ToTensor()
            self.train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
            self.test_dataset = CIFAR100(root="./data", train=False, download=True, transform=transform)
            total_classes = 100
        elif self.dataset_name == 'cifar10':
            transform = transforms.ToTensor()
            self.train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
            self.test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
            total_classes = 10
        elif self.dataset_name == 'fashionmnist':
            # FashionMNIST is grayscale, convert to RGB for CLIP
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
            self.train_dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
            self.test_dataset = FashionMNIST(root="./data", train=False, download=True, transform=transform)
            total_classes = 10
        elif self.dataset_name == 'stl10':
            transform = transforms.ToTensor()
            self.train_dataset = STL10(root="./data", split='train', download=True, transform=transform)
            self.test_dataset = STL10(root="./data", split='test', download=True, transform=transform)
            # STL-10 uses 'labels' instead of 'targets', we'll handle this in get_task_datasets
            total_classes = 10
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' is not supported.")
            
        if total_classes % self.num_tasks != 0:
            raise ValueError(f"Total classes ({total_classes}) must be divisible by num_tasks ({self.num_tasks}).")
        
        self.classes_per_task = total_classes // self.num_tasks
        self.class_names = self.train_dataset.classes

    def get_task_datasets(self, dataset_type, task_id):
        """
        Gets a subset of the data corresponding to a specific task.
        """
        dataset = self.train_dataset if dataset_type == 'train' else self.test_dataset
        start_class = task_id * self.classes_per_task
        end_class = (task_id + 1) * self.classes_per_task
        
        # STL-10 uses 'labels' instead of 'targets'
        if self.dataset_name == 'stl10':
            targets = dataset.labels
        else:
            targets = dataset.targets
        
        indices = [i for i, target in enumerate(targets) if start_class <= target < end_class]
        return Subset(dataset, indices)
        
    def get_class_range_for_task(self, task_id):
        """Returns the start and end class ID for a given task."""
        start_class = task_id * self.classes_per_task
        end_class = (task_id + 1) * self.classes_per_task
        return range(start_class, end_class)

    def get_class_names_for_task(self, task_id):
        """Returns the list of class name strings for a given task."""
        if not self.class_names:
             raise RuntimeError("Class names have not been loaded. Call _load_dataset first.")
        class_range = self.get_class_range_for_task(task_id)
        return [self.class_names[i] for i in class_range]
