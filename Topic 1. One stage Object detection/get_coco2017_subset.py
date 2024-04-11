import os

import fiftyone as fo
import fiftyone.zoo as foz


if __name__ == "__main__":
    dataset_subset_dir = "dataset_subset"
    subsets_amount = {"train": 500, "validation": 100}
    classes = ["person", "car", "cat"]
    label_types = ["detections"]

    for subset in subsets_amount:
        n_samples = subsets_amount[subset]

        # Load n_samples random samples from the validation split
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split=subset,
            max_samples=n_samples,
            shuffle=True,
            label_types=label_types,
            classes=classes,
        )

        # Export the dataset
        export_dir = os.path.join(dataset_subset_dir, subset)
        dataset.export(
            export_dir=export_dir, dataset_type=fo.types.COCODetectionDataset
        )
