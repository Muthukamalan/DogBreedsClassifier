from typing import Optional, AnyStr
import os
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DogsBreadDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        data_dir: Optional[AnyStr] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.prepare_data()

    def prepare_data(self) -> None:
        if os.path.isdir(self.hparams.data_dir) == False:
            # download and parse dataset
            """
            dogs_dataset/
                    ├── test
                    │   ├── Beagle
                    │   ├── Boxer
                    │   ├── Bulldog
                    │   ├── Dachshund
                    │   ├── German_Shepherd
                    │   ├── Golden_Retriever
                    │   ├── Labrador_Retriever
                    │   ├── Poodle
                    │   ├── Rottweiler
                    │   └── Yorkshire_Terrier
                    ├── train
                    │   ├── Beagle
                    │   ├── Boxer
                    │   ├── Bulldog
                    │   ├── Dachshund
                    │   ├── German_Shepherd
                    │   ├── Golden_Retriever
                    │   ├── Labrador_Retriever
                    │   ├── Poodle
                    │   ├── Rottweiler
                    │   └── Yorkshire_Terrier
                    └── validation
                        ├── Beagle
                        ├── Boxer
                        ├── Bulldog
                        ├── Dachshund
                        ├── German_Shepherd
                        ├── Golden_Retriever
                        ├── Labrador_Retriever
                        ├── Poodle
                        ├── Rottweiler
                        └── Yorkshire_Terrier
            """
            from dotenv import load_dotenv

            print(
                f"loaded config:: {load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))}"
            )
            os.environ['KAGGLE_USERNAME'] = os.environ['username']
            os.environ['KAGGLE_KEY'] = os.environ['key']

            # after loading kaggle key then imported
            import kaggle
            import random
            import shutil

            kaggle.api.authenticate()
            dogs_dataset = self.hparams.data_dir
            kaggle.api.dataset_download_files(
                'khushikhushikhushi/dog-breed-image-dataset',
                path=dogs_dataset,
                unzip=True,
            )

            base_dir = os.path.join(dogs_dataset, 'dataset')

            train_dir = os.path.join(dogs_dataset, 'train')
            val_dir = os.path.join(dogs_dataset, 'validation')
            test_dir = os.path.join(dogs_dataset, 'test')
            breed_folders = [
                f
                for f in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, f))
            ]

            for split_dir in [train_dir, val_dir, test_dir]:
                os.makedirs(split_dir, exist_ok=True)
            train_ratio = 0.7  # 70% of images for training
            val_ratio = 0.1  # 20% of images for validation
            test_ratio = 0.1  # 10% of images for testing

            for breed in breed_folders:
                breed_path = os.path.join(base_dir, breed)

                os.makedirs(os.path.join(train_dir, breed), exist_ok=True)
                os.makedirs(os.path.join(val_dir, breed), exist_ok=True)
                os.makedirs(os.path.join(test_dir, breed), exist_ok=True)

                images = [
                    f
                    for f in os.listdir(breed_path)
                    if os.path.isfile(os.path.join(breed_path, f))
                ]

                random.shuffle(images)

                total_images = len(images)
                train_count = int(total_images * train_ratio)
                val_count = int(total_images * val_ratio)
                test_count = total_images - train_count - val_count

                train_images = images[:train_count]
                val_images = images[train_count : train_count + val_count]
                test_images = images[train_count + val_count :]

                for image in train_images:
                    shutil.move(
                        os.path.join(breed_path, image),
                        os.path.join(train_dir, breed, image),
                    )

                # Move images to val directory
                for image in val_images:
                    shutil.move(
                        os.path.join(breed_path, image),
                        os.path.join(val_dir, breed, image),
                    )

                # Move images to test directory
                for image in test_images:
                    shutil.move(
                        os.path.join(breed_path, image),
                        os.path.join(test_dir, breed, image),
                    )

                if (
                    breed_path.endswith('test')
                    or breed_path.endswith('train')
                    or breed_path.endswith('validation')
                ):
                    pass
                else:
                    shutil.rmtree(breed_path)
            shutil.rmtree(base_dir)
            print('Dataset Downloadded and Extracted Successfully!')

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageFolder(
                root=os.path.join(str(self.hparams.data_dir), 'train'),
                transform=self.transforms,
            )
            self.test_dataset = ImageFolder(
                root=os.path.join(str(self.hparams.data_dir), 'test'),
                transform=self.transforms,
            )
            self.validation_dataset = ImageFolder(
                root=os.path.join(str(self.hparams.data_dir), 'validation'),
                transform=self.transforms,
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )
