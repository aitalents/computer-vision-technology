import numpy as np
import torch

def nms(pred, threshold=0.5):

    with torch.no_grad():
        pred = pred.reshape((-1, 30))
        nms_data = [[] for _ in range(20)]
        for i in range(pred.shape[0]):
            cell = pred[i]
            score, idx = torch.max(cell[10:30], dim=0)
            idx = idx.item()
            x, y, w, h, iou = cell[0:5].cpu().numpy()

            nms_data[idx].append([i, x, y, w, h, iou, score.item()])
            x, y, w, h, iou = cell[5:10].cpu().numpy()
            nms_data[idx].append([i, x, y, w, h, iou, score.item()])

        ret = torch.zeros_like(pred)
        flag = torch.zeros(pred.shape[0], dtype=torch.bool)
        for c in range(20):
            c_nms_data = np.array(nms_data[c])

            keep_index = _nms(c_nms_data, threshold)
            keeps = c_nms_data[keep_index]

            for keep in keeps:
                i, x, y, w, h, iou, score = keep
                i = int(i)

                last_score, _ = torch.max(ret[i][10:30], dim=0)
                last_iou = ret[i][4]

                if score * iou > last_score * last_iou:
                    flag[i] = False
                if flag[i]: continue

                ret[i][0:5] = torch.tensor([x, y, w, h, iou])
                ret[i][10:30] = 0
                ret[i][10 + c] = score

                flag[i] = True

        return ret
    
    
def _nms(data, threshold):

    if len(data) == 0:
        return []

    cell_idx = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    xidx = cell_idx % 7
    yidx = cell_idx // 7
    x = (x + xidx) / 7.0
    y = (y + yidx) / 7.0
    w = data[:, 3]
    h = data[:, 4]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    score_area = data[:, 5]

    areas = w * h

    order = score_area.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep
