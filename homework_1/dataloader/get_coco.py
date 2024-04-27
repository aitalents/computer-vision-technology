# https://docs.voxel51.com/user_guide/app.html
import os
import fiftyone as fo
import fiftyone.zoo as foz
import torch
from PIL import Image
from .transform import Transformtsz, FiftyOneTorchDataset

""" Full split stats

    Train split: 118,287 images

    Test split: 40,670 images

    Validation split: 5,000 images """

# presuming that cwd is CV_HW_1
cv_dir = os.getcwd()
data_dir = os.path.join(cv_dir, "data")
fo.config.dataset_zoo_dir = data_dir

# выберите sample сами, если, весь датасет -144.1 GB + 1.9 GB, грузить его так, весь - пиздец, но если у кого-то есть возможность - было бы здорово # noqa

def load_coco(max_samples):
    dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits = ["train", "validation", "test"],
    label_types = ["detections"],
    # classes = classes
    max_samples = max_samples,
    dataset_dir="../data")
    
    dataset.compute_metadata()
    return dataset

'''classes = dataset.distinct(
    "ground_truth.detections.label"
)
id2name, name2id = {}, {}
classes_id_list = []
for class_id, class_name in enumerate(classes):
    if class_name in classes:
      classes_id_list.append(class_id)
    id2name[class_id] = class_name
    name2id[class_name] = class_id'''


def ttsplit(dataset):
    train_data = dataset.match_tags("train")
    test_data = dataset.match_tags("test")
    val_data = dataset.match_tags("validation")
    return train_data, test_data, val_data

def get_torch(dataset):
    classes = dataset.distinct(
    "ground_truth.detections.label"
    )
    torch_dataset = FiftyOneTorchDataset(dataset, transforms=Transformtsz(resize=(448, 448)), classes=classes)
    return dataset

def get_loader(torch_dataset):
    data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=1, shuffle=False)
    return data_loader

if __name__ == "__main__":
    load_coco(5000)