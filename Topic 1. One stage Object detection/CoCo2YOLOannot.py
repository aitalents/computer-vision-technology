import json
import os


def convert_coco_to_yolo(coco_annotations_file, images_dir, output_dir, class_map=None):

    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    image_id_to_annotations = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(annotation)

    for image in images:
        image_id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']


        if class_map:
            classes = {cat['id']: idx for idx, cat in enumerate(categories)}
        else:
            classes = {cat['id']: cat['name'] for cat in categories}

        annotations = image_id_to_annotations.get(image_id, [])
        lines = []
        for annotation in annotations:
            category_id = annotation['category_id']
            category = classes[category_id]
            bbox = annotation['bbox']
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            x_center /= width
            y_center /= height
            bbox_width = bbox[2] / width
            bbox_height = bbox[3] / height
            if class_map:
                category_id = class_map.get(category_id, -1)
                if category_id == -1:
                    continue
            lines.append(f"{category_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")



        if lines:
            output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.txt')
            with open(output_file, 'w') as f:
                f.writelines(lines)


if __name__ == "__main__":
    coco_annotations_file = "./data/annotations/instances_val2017.json"
    images_dir = "./data/val2017"
    output_dir = "./data/annotations/val_annot"

    # Optionally, provide a class mapping if you want to filter out some classes
    # class_map = {1: 0, 2: 1, 3: 2}  # Example class mapping: maps COCO category ids to YOLO class ids
    some_list = [44, 67, 1, 49, 51, 79, 47, 56, 50, 57, 81, 64, 62, 82, 52, 55, 28, 31, 10, 2, 41, 3, 8, 70, 4, 16, 76,
                 84, 72, 86, 63, 5, 33, 25, 21, 9, 15, 20, 6, 27, 7, 13, 18, 17, 73, 32, 22, 85, 34, 23, 24, 19, 35, 37,
                 40, 60, 54, 61, 42, 65, 59, 43, 90, 75, 53, 36, 38, 39, 11, 74, 88, 77, 87, 46, 48, 78, 58, 14, 80, 89]
    some_list = sorted(some_list)
    class_map = {some_list[i] : i for i in range(len(some_list))}

    convert_coco_to_yolo(coco_annotations_file, images_dir, output_dir, class_map)
