def is_intersect(self, boxA, boxB):
    if boxA[0] > boxB[2]:
        return False  # boxA is right of boxB
    if boxA[1] > boxB[3]:
        return False  # boxA is below boxB
    if boxA[2] < boxB[0]:
        return False  # boxA is left boxB
    if boxA[3] < boxB[1]:
        return False  # boxA is above boxB
    return True


def get_intersection(self, boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return (xB - xA + 1) * (yB - yA + 1)


def get_union(self, boxA, boxB):
    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return area_A + area_B


def transform_x1y1wh_to_x1y1x2y2(self, box):
    x1 = round(box[0], 2)
    y1 = round(box[1], 2)
    x2 = round(box[0] + box[2], 2)
    y2 = round(box[1] + box[3], 2)
    return [x1, y1, x2, y2]


def get_IoU(self, boxA, boxB):
    # x1y1wh -> x1y1x2y2
    boxA = self.transform_x1y1wh_to_x1y1x2y2(boxA)
    boxB = self.transform_x1y1wh_to_x1y1x2y2(boxB)
    if self.is_intersect(boxA, boxB) is False:
        return 0
    inter = self.get_intersection(boxA, boxB)
    union = self.get_union(boxA, boxB)
    iou = inter / (union - inter)
    return iou
