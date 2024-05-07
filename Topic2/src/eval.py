def id_f1(preds, true):
    tp, fp, fn = 0, 0, 0
    for i in range(len(preds)):
        for j in range(len(preds[i]['data'])):
            if len(preds[i]['data'][j]['bounding_box']) == 0:
                continue
            if not preds[i]['data'][j]['track_id']:
                fn += 1
            elif preds[i]['data'][j]['track_id'] == true[i]['data'][j]['cb_id']:
                tp += 1
            else:
                fp += 1
    return 2 * tp / (2 * tp + fp + fn)