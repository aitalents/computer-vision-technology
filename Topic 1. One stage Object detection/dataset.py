import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO


class COCODetDataset(Dataset):
    def __init__(self, data_folder, img_size=(448,448)):
        self.resize_to = img_size
        
        if not os.path.exists(data_folder):
            raise ValueError(f'Data folder does not exist! Input path: {data_folder}')
        
        self.imgs_folder = os.path.join(data_folder, 'data')
        self.markup_json = os.path.join(data_folder, 'labels.json')

        self.coco = COCO(self.markup_json)
        self.img_ids = sorted(self.coco.getImgIds())

        self.class_names, self.class_ids = self.get_class_names()

    def __len__(self):
        return len(self.img_ids)
    
    def load_image(self, img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(self.resize_to)

        return transforms.ToTensor()(img)
    
    def get_gt_bboxes_whole_image(self, anns, img_markup):
        """
        Input format: COCO bboxes in format (x_min, y_min, w, h, class_id)
        Output format: (x_center, y_center, w, h, class_id) and normalized coordinates
        """
        gt_bboxes = []
        img_width = img_markup['width']
        img_height = img_markup['height']

        for ann in anns:
            if 'bbox' in ann and 'category_id' in ann:
                x_abs, y_abs, w_abs, h_abs = ann['bbox']
                class_id = float(self.class_ids.index(ann['category_id']))
                x_c, y_c = (x_abs + w_abs * 0.5) / img_width, (y_abs + h_abs * 0.5) / img_height
                w, h = w_abs / img_width, h_abs / img_height

                gt_bboxes.append([x_c, y_c, w, h, class_id])

        return torch.Tensor(gt_bboxes)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_markup = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.imgs_folder, img_markup['file_name'])
        img = self.load_image(img_path)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        gt_bboxes = self.get_gt_bboxes_whole_image(anns, img_markup)

        return img, gt_bboxes
    
    def get_class_names(self):
        cat_info = self.coco.dataset['categories']
        class_names = [class_info['name'] for class_info in cat_info]
        class_ids = [class_info['id'] for class_info in cat_info]

        return class_names, class_ids
    

def batch_collate_fn(batch, grid_size=7):
    """
    For each gt detection build matrix with shape (grid_size, grid_size, 5),
    then concatenate all these matrices.

    Into matrix save information about what bboxes were in every cell in grid.
    ! Important note: only one (or none) gt bbox could be in matrix for each cell!
    """
    images = [item[0].unsqueeze(0) for item in batch]
    detections = []

    for item in batch:
        dets = item[1]
        image_detections = torch.zeros(1, grid_size, grid_size, 5)

        for det in dets:      
            cell_col = int(grid_size * det[0])
            cell_row = int(grid_size * det[1])
            image_detections[0, cell_col, cell_row, :4] = det[:4]
            image_detections[0, cell_col, cell_row, -1] = det[-1]   
             
        detections.append(image_detections)

    images = torch.cat(images, 0)
    detections = torch.cat(detections, 0)    

    return (images, detections)


if __name__ == '__main__':
    coco_dataset_path = r'..\..\..\..\..\..\fiftyone\coco-2017'

    train_data_folder = os.path.join(coco_dataset_path, 'train')
    val_data_folder = os.path.join(coco_dataset_path, 'validation')

    train_dataset = COCODetDataset(train_data_folder)
    val_dataset = COCODetDataset(val_data_folder)

    # example
    img, dets = train_dataset[0]
    print(f'first train img tensor: {img}')
    print(f'first train gt boxes: {dets}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=2, shuffle=True,
        collate_fn=batch_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2, shuffle=False,
        drop_last=False,
        collate_fn=batch_collate_fn,
    )

    for batch in train_loader:
        imgs, gt_dets = batch
        print(f'first batch dets: {gt_dets}')
        break
