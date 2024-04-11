import os
import csv


def generate_csv(images_dir, annotations_dir, output_csv):
    image_files = [file for file in os.listdir(images_dir) if file.endswith('.jpg') or file.endswith('.png')]
    annotation_files = [file for file in os.listdir(annotations_dir) if file.endswith('.txt')]

    image_annotation_pairs = []
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        annotation_file = image_name + '.txt'
        if annotation_file in annotation_files:
            image_path = image_file
            annotation_path = annotation_file
            image_annotation_pairs.append((image_path, annotation_path))

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'annotation_path'])
        for pair in image_annotation_pairs:
            writer.writerow(pair)


if __name__ == "__main__":
    images_dir = "./data/train2017"
    annotations_dir = "./data/annotations/train_annot"
    output_csv = "./data/annotations/train.csv"

    generate_csv(images_dir, annotations_dir, output_csv)
