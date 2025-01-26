import os
from enum import Enum
import PIL
import torch
from torchvision import transforms

# List of datasets available in Viaduct
DATASETS = [
    "05_cable_end_sleeve",
    '06_coaxial_t-adapter',
    # Add more datasets here if applicable
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ViaductDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Viaduct.
    """

    def __init__(
        self,
        source,
        dataset_name,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        rotate_degrees=0,
        translate=0,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        scale=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the Viaduct data folder.
            dataset_name: [str]. Name of the Viaduct dataset that should be provided.
            resize: [int]. (Square) Size the loaded image initially gets resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets (center-)cropped to.
            split: [enum-option]. Indicates if training, validation, or test split should be used.
        """
        self.dataset_name = dataset_name
        super().__init__()
        self.source = source
        self.split = split
        self.train_val_split = train_val_split
        self.imagesize = (3, imagesize, imagesize)

        # Image transformation pipeline
        self.transform_img = transforms.Compose([
            transforms.Resize(resize),  # Resize the image to 'resize' dimension (256)
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(rotate_degrees,
                                    translate=(translate, translate),
                                    scale=(1.0 - scale, 1.0 + scale),
                                    interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imagesize),  # Crop to the desired image size (224)
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Mask transformation pipeline
        self.transform_mask = transforms.Compose([
            transforms.Resize(resize),  # Resize the mask to 'resize' dimension (256)
            transforms.CenterCrop(imagesize),  # Crop to the desired mask size (224)
            transforms.ToTensor(),
        ])

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

    def __getitem__(self, idx):
        anomaly, image_path, mask_path = self.data_to_iterate[idx]

        # Load and transform image
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # Load and transform mask (for test split, mask is used)
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            # If not a test image, use an empty mask (zeros tensor of correct shape)
            mask = torch.zeros([1, self.imagesize, self.imagesize])

        # Return the data dictionary
        return {
            "image": image,
            "mask": mask,
            "dataset_name": self.dataset_name,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        dataset_path = os.path.join(self.source, self.dataset_name, self.split.value)
        mask_path = os.path.join(self.source, self.dataset_name, "ground_truth")
        anomaly_types = os.listdir(dataset_path)

        for anomaly in anomaly_types:
            anomaly_dir = os.path.join(dataset_path, anomaly)
            if not os.path.isdir(anomaly_dir):
                continue

            anomaly_files = sorted(os.listdir(anomaly_dir))
            imgpaths_per_class[anomaly] = [os.path.join(anomaly_dir, x) for x in anomaly_files]

            if self.train_val_split < 1.0:
                n_images = len(imgpaths_per_class[anomaly])
                split_idx = int(n_images * self.train_val_split)
                if self.split == DatasetSplit.TRAIN:
                    imgpaths_per_class[anomaly] = imgpaths_per_class[anomaly][:split_idx]
                elif self.split == DatasetSplit.VAL:
                    imgpaths_per_class[anomaly] = imgpaths_per_class[anomaly][split_idx:]

            if self.split == DatasetSplit.TEST and anomaly != "good":
                mask_dir = os.path.join(mask_path, anomaly)
                mask_files = sorted(os.listdir(mask_dir))
                maskpaths_per_class[anomaly] = [os.path.join(mask_dir, x) for x in mask_files]
            else:
                maskpaths_per_class["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for anomaly in sorted(imgpaths_per_class.keys()):
            for i, image_path in enumerate(imgpaths_per_class[anomaly]):
                mask_path = (
                    maskpaths_per_class[anomaly][i]
                    if self.split == DatasetSplit.TEST and anomaly != "good"
                    else None
                )
                data_to_iterate.append((anomaly, image_path, mask_path))

        return imgpaths_per_class, data_to_iterate
