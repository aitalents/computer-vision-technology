import os

import pandas as pd


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


def preprocess(
    files_dir="./train2017", test_dir="./val2017", annotations="./annotations"
):
    images = [image for image in sorted(os.listdir(files_dir)) if image[-4:] == ".jpg"]
    annots = []
    for image in images:
        annot = image[:-4] + ".xml"
        annots.append(annot)

    images, annots = pd.Series(images, name="images"), pd.Series(annots, name="annots")
    df = pd.DataFrame(pd.concat([images, annots], axis=1))

    test_images = [
        image for image in sorted(os.listdir(test_dir)) if image[-4:] == ".jpg"
    ]

    test_annots = []
    for image in test_images:
        annot = image[:-4] + ".xml"
        test_annots.append(annot)

    test_images, test_annots = (
        pd.Series(test_images, name="test_images"),
        pd.Series(test_annots, name="test_annots"),
    )
    test_df = pd.DataFrame(pd.concat([test_images, test_annots], axis=1))
    test_df = pd.DataFrame(test_df)
    return df, test_df
