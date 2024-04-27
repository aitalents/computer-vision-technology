import torch
import torchvision
import fiftyone
from PIL import Image
import fiftyone.utils.coco as fouc
from itertools import product
import json

class FiftyOneTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """

    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        boxes = []
        labels = []
        area = []
        iscrowd = []
        if sample[self.gt_field] is not None:
            detections = sample[self.gt_field].detections
            for det in detections:
                category_id = self.labels_map_rev[det.label]
                coco_obj = fouc.COCOObject.from_label(
                    det, metadata, category_id=category_id,
                )
                x, y, w, h = coco_obj.bbox
                boxes.append([(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]) # normalized (xc, yc, w, h)
                labels.append(coco_obj.category_id)
                area.append(coco_obj.area)
                iscrowd.append(coco_obj.iscrowd)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

class Transformtsz:
    def __init__(self, resize):
        self.resize = resize
    def __call__(self, image, boxes):
        image = torchvision.transforms.functional.resize(image, self.resize)
        image = torchvision.transforms.functional.to_tensor(image)
        return image, boxes

def collate(batch, grid_size=7, n_classes=80):
    images = []
    gt = []
    for item in batch:
        images.append(item[0].unsqueeze(0))

        fmap = torch.zeros(1, grid_size, grid_size, 5*2+n_classes)
        bboxes = item[1]["boxes"]
        labels = item[1]["labels"]

        used_col_row = {(r, c): 0 for r,c in list(product(range(7), repeat=2))}
        for bbox, label in zip(bboxes, labels):
            col = int(bbox[1] * grid_size)
            row = int(bbox[0] * grid_size)
            cell_size = 1 / grid_size
            row_interval = (cell_size*row, cell_size*(row+1))
            col_interval = (cell_size*col, cell_size*(col+1))

            # if more than 2 bboxes in one cell then skip
            if used_col_row[(row, col)] == 2:
                continue

            used_col_row[(row, col)] += 1

            if used_col_row[(row, col)] == 1:
                # bbox center coords relative to grid cell
                fmap[0, row, col, 0]  = (bbox[0] - row_interval[0]) / (row_interval[1] - row_interval[0])
                fmap[0, row, col, 1]  = (bbox[1] - col_interval[0]) / (col_interval[1] - col_interval[0])
                fmap[0, row, col, 2:4]  = bbox[2:] # bbox w and h relative to image size
                fmap[0, row, col, 4] = 1 # confindece
            elif used_col_row[(row, col)] == 2:
                # bbox center coords relative to grid cell
                fmap[0, row, col, 5]  = (bbox[0] - row_interval[0]) / (row_interval[1] - row_interval[0])
                fmap[0, row, col, 6]  = (bbox[1] - col_interval[0]) / (col_interval[1] - col_interval[0])
                fmap[0, row, col, 7:9]  = bbox[2:] # bbox w and h relative to image size
                fmap[0, row, col, 9] = 1 # confindece
            # set classes probabilities
            fmap[0, row, col, label - 1 + 10] = 1
        gt.append(fmap)
    
    images = torch.cat(images, 0)
    detections = torch.cat(gt, 0)
    return (images, detections)

class RawDataforTest(torch.utils.data.Dataset):
    
    def __init__(self, fiftyone_dataset, transforms=None, gt_field="ground_truth", classes=None,):
        
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.detections.label" % gt_field
            )

        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        metadata = sample.metadata
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        if sample[self.gt_field] is not None:
            detections = sample[self.gt_field].detections
            for det in detections:
                category_id = self.labels_map_rev[det.label]
                coco_obj = fouc.COCOObject.from_label(
                    det, metadata, category_id=category_id,
                )   
                x, y, w, h = coco_obj.bbox
                
                # update labels from absolute to relative
                h, w = float(h), float(w)

                ret_targets = []
                ret_targets.append({
                        'xmin': float(x) / w,
                        'ymin': float(y) / h,
                        'xmax': float(x+w) / w,
                        'ymax': float(x+h) / h,
                        'category': category_id,
                })
                
                
            img = torchvision.transforms.functional.resize(img, (448, 448))
            #img = torchvision.transforms.ToTensor(img)
            img = torchvision.transforms.functional.to_tensor(img)
            #if self.transforms is not None:
                #img, target = self.transforms(img, target)
        
            return img, json.dumps(ret_targets)
