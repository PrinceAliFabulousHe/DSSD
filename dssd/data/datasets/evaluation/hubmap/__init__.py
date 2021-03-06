import numpy as np
from scipy.optimize import linear_sum_assignment
import logging


def intersection(a, b):
    shape_a = a.shape[0]
    shape_b = b.shape[0]
    intersections_min = np.maximum(np.tile(np.expand_dims(a[..., [0, 1]], axis=1), reps=(1, shape_b, 1)),
                                   np.tile(np.expand_dims(b[..., [0, 1]], axis=0), reps=(shape_a, 1, 1)))
    intersections_max = np.minimum(np.tile(np.expand_dims(a[..., [2, 3]], axis=1), reps=(1, shape_b, 1)),
                                   np.tile(np.expand_dims(b[..., [2, 3]], axis=0), reps=(shape_a, 1, 1)))

    side_lengths = np.maximum(0, intersections_max - intersections_min)
    return side_lengths[..., 0] * side_lengths[..., 1]


def iou(a, b):
    intersections = intersection(a, b)
    areas_1 = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    areas_2 = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    shape_a = a.shape[0]
    shape_b = b.shape[0]
    areas_1 = np.tile(np.expand_dims(areas_1, axis=1), reps=(1, shape_b))
    areas_2 = np.tile(np.expand_dims(areas_2, axis=0), reps=(shape_a, 1))
    return intersections / (areas_1 + areas_2 - intersections + 1e-15)


def predictions2labels(x, y, iou_th=0.5, conf_th=0.5):
    # make a new prediction only if there is no stored
    result = []
    for i, prediction in enumerate(x):
        prediction = prediction[prediction[:, 0] >= conf_th]
        # if there are no prediction skip because only true/false positive are interesting
        if prediction.size == 0:
            result.append(np.array([[-1, -1]]))
            continue
        # get labels for the current image from the dataset
        labels = y[i]
        # if there are no labels on the image set all predictions to false positive
        if labels.size == 0:
            result.append(np.vstack([prediction[..., 0], np.zeros((prediction.shape[0],))]).T)
            continue
        elif labels.shape[1] == 5:
            print("Warning: Labels has last axis of size 5. Probably there is an score inside. "
                  "First value is removed!")
            labels = labels[:, 1:]
        # calculate the IoU between every predicted and every ground truth box
        overlap = iou(prediction[..., 1:], labels)

        # set the overlap to zero, which is smaller then the threshold
        overlap_filtered = overlap * (overlap > iou_th)
        # match every predicted box to exact one label
        match_pred, match_gt = linear_sum_assignment(overlap, maximize=True)
        # set all boxes to fp
        tp_fp = np.zeros((prediction.shape[0],))
        # every box, which has been matched to a value higher than zero, is a true positive prediction. Otherwise
        # its a false positive, because there is no label
        satisfy_thresh = np.where(overlap_filtered[match_pred, match_gt] > 0.0, True, False)
        # create a mask over all predictions whether they are tp
        mask = match_pred[satisfy_thresh]
        # set the tp to one. The masking step is necessary because the there can be more predicted boxes than labels
        tp_fp[mask] = 1
        # extract the scores from the predictions
        scores = prediction[..., 0]
        # merge the score with the evaluation
        result.append(np.vstack([scores, tp_fp]).T)
    return result


def flat_precision_recall(assigned_predictions, gt_boxes):
    nr_gt_labels = np.concatenate(gt_boxes, axis=0).shape[0]
    merged_predictions = np.concatenate(assigned_predictions, axis=0)
    tp = np.sum(merged_predictions[..., 1] == 1)
    fp = merged_predictions.shape[0] - tp
    precision = tp / (tp + fp)
    recall = tp / nr_gt_labels  # number of ground truth is tp + fn
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
    return precision, recall, f1


def precision_at_recall(assigned_prediction, gt_boxes):
    nr_gt_labels = np.concatenate(gt_boxes, axis=0).shape[0]
    # merge the matched predictions of all images to one array
    merged_predictions = np.concatenate(assigned_prediction, axis=0)
    # sort predictions by highest confidence first
    ordered_prediction = merged_predictions[merged_predictions[..., 0].argsort()][::-1]
    # remove scores from predictions
    c = ordered_prediction[..., 1]
    cum_tp = np.cumsum(c)
    cum_fp = np.cumsum(c == 0)
    precision = np.where(cum_tp + cum_fp > 0, cum_tp / (cum_tp + cum_fp), 0)
    recall = cum_tp / nr_gt_labels

    aP = np.trapz(precision, recall)
    return precision, recall, aP


def hubmap_evaluation(dataset, predictions, output_dir, iteration=None):
    predicted_boxes = []
    gt_boxes = []
    for i, prediction in enumerate(predictions):
        prediction = prediction.numpy()
        boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']

        pred_score_boxes = np.hstack([np.expand_dims(scores, -1), boxes])
        predicted_boxes.append(pred_score_boxes)

        img, target, index = dataset[i]
        gt_boxes.append(target['boxes'])

    assigned = predictions2labels(predicted_boxes, gt_boxes, iou_th=0.5, conf_th=0.01)
    acc_precision, acc_recall, aP = precision_at_recall(assigned, gt_boxes)
    assigned = predictions2labels(predicted_boxes, gt_boxes, iou_th=0.5, conf_th=0.5)
    precision, recall, f1 = flat_precision_recall(assigned, gt_boxes)

    logger = logging.getLogger("DSSD.inference")
    logger.info('Validation: mAP: {0:.3f} Precision: {1:.3f} Recall: {2:.3f} F1: {0:.3f}'.format(aP, precision, recall, f1))
    metric = {"mAP@0.5": round(aP, 3),
              "Precision": round(precision, 3),
              "Recall": round(recall, 3),
              "F1": round(f1, 3)}
    return dict(metrics=metric)
