import numpy as np


seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat
class_labels = {c: l for l, c in enumerate(sorted(seg_classes.keys()))}
part_codes = []
for k in sorted(seg_classes.keys()): part_codes += [seg_classes[k]]

def get_evaluation_metrics(logits, labels):

    seg = np.ones_like(labels)*(-1)
    shape_IoUs = {c: [] for c in seg_classes.keys()}
    for i, (l, y) in enumerate(zip(logits, labels)):
        y = y.reshape(-1)
        cls_parts = seg_classes[seg_label_to_cat[y[0]]]
        category = cls_parts[0]

        # Point predictions
        s = l[:, cls_parts].argmax(-1) + category

        # Find IoU for each part in the point cloud
        part_IoUs = []
        for p in cls_parts:
            s_p, y_p = (s == p), (y == p)
            iou = (s_p & y_p).sum() / float((s_p | y_p).sum()) if np.any(s_p | s_p) else 1.0
            part_IoUs += [iou]
        
        seg[i] = s
        shape_IoUs[seg_label_to_cat[category]] += [np.mean(part_IoUs)]

    # Overall point accuracy
    acc = (seg == labels).sum() / np.prod(labels.shape)

    class_accs = []
    for i in range(len(np.unique(labels))):
        labels_i = (labels == i)
        seg_i = (seg == i)
        class_accs.append((labels_i & seg_i).sum() / labels_i.sum())
    
    # Mean class accuracy (point-wise)
    mean_class_accuracy = np.mean(class_accs)

    mean_shape_IoUs = []
    instance_IoUs = []
    for c in shape_IoUs.keys():        
        instance_IoUs += shape_IoUs[c]
        mean_shape_IoUs += [np.mean(shape_IoUs[c])]

    # Overall IoU on all samples
    average_instance_IoUs = np.mean(instance_IoUs)

    # Mean class IoU: average IoUs of (Airplane, bag, cap, ..., table)
    average_shape_IoUs = np.mean(mean_shape_IoUs)

    summary = {}
    summary["acc"] = acc 
    summary["mean_class_accuracy"] = mean_class_accuracy
    summary["average_instance_IoUs"] = average_instance_IoUs
    summary["average_shape_IoUs"] = average_shape_IoUs
    summary["shape_IoUs"] = {k: v for k, v in zip(seg_classes.keys(), mean_shape_IoUs)}

    return summary


def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred = pred.max(dim=2)[1]    # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

            F = np.sum(target_np[shape_idx] == part)

            if F != 0:
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]
