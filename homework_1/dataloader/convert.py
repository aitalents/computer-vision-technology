import torch
import numpy as np


def convert_label_matrix_to_bboxes(label_matrix, S=7, C=5, B=2, img_size=448):
    height, width, _ = label_matrix.shape

    bboxes = []
    predicted_classes = []
    confidence = []

    for i in range(height):
        for j in range(width):
            cell_label = label_matrix[i, j]

            if cell_label[C] != 0:
                cell_x, cell_y, width_cell, height_cell = cell_label[C+1:C+5]

                xmin = (j + cell_x - width_cell / 2).item() / S * img_size
                ymin = (i + cell_y - height_cell / 2).item() / S * img_size
                xmax = (j + cell_x + width_cell / 2).item() / S * img_size
                ymax = (i + cell_y + height_cell / 2).item() / S * img_size

                bboxes.append([xmin, ymin, xmax, ymax])
                predicted_classes.append(torch.argmax(cell_label[0:C]).item())
                confidence.append(cell_label[C])

    return np.array(bboxes), np.array(predicted_classes), np.array(confidence)

  
