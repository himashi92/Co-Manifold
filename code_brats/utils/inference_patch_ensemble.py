import pprint
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from monai.data import decollate_batch
from .monai_utils import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations, KeepLargestConnectedComponent
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def pad_batch1_to_compatible_size(batch):
    print(batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
    zmax, ymax, xmax = zyx
    zpad, ypad, xpad = zmax - batch.size(2), ymax - batch.size(3), xmax - batch.size(4)
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    return batch, (zpad, ypad, xpad)


post_trans = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)
device = torch.device("cuda:0")
VAL_AMP = True


# define inference method
def inference(input, model, patch_size):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.85
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


def calculate_metrics(preds, targets, patient, tta=False):
    """
    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []
    dice_total = 0.0
    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )

        if np.sum(targets[i]) == 0:
            print(f"{label} not present for {patient}")
            iou = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            asd = np.nan
            haussdorf_dist = np.nan

        else:
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            iou = tp / (tp + fp + fn)
            asd = tn / (tn + fp)

            dice = 2 * tp / (2 * tp + fp + fn)
            dice_total += dice

        metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[IOU] = iou
        metrics[ASD] = asd
        pp.pprint(metrics)
        metrics_list.append(metrics)
        dice_avg = dice_total / 3

    return metrics_list, dice_avg


def test_all_case(model1, model2, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, label_batch, seg_path, crops_idx = batch['image'], batch['label'], batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_labels = volume_batch.cuda(), label_batch.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            with torch.no_grad():
                val_outputs_1_1 = inference(val_inputs, model1, patch_size)

                val_outputs_1_2 = inference(val_inputs, model2, patch_size)

                y = torch.zeros(val_outputs_1_1.shape).cuda()
                y = val_outputs_1_1 + val_outputs_1_2

                y /= 2
            val_outputs = [post_trans(i) for i in decollate_batch(y)]
            # val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs_1_1)]
            # val_outputs_2 = [post_trans(i) for i in decollate_batch(val_outputs_1_2)]

            # val_outputs_1_argmax = [AsDiscrete(argmax=True)(i) for i in decollate_batch(val_outputs_1_1)]
            # val_outputs_2_argmax = [AsDiscrete(argmax=True)(i) for i in decollate_batch(val_outputs_1_2)]
            #
            # val_outputs_1_largest = [KeepLargestConnectedComponent(applied_labels=[1])(i) for i in val_outputs_1_argmax]
            # val_outputs_2_largest = [KeepLargestConnectedComponent(applied_labels=[1])(i) for i in val_outputs_2_argmax]
            #
            # print(torch.unique(val_outputs_1_largest[0]))
            # print(torch.unique(val_outputs_2_largest[0]))
            # val_outputs = torch.logical_and(val_outputs_1_largest[0], val_outputs_2_largest[0])

            segs = torch.zeros((1, 3, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs[0]
            segs = segs[0].numpy() > 0.5

            et = segs[0]
            net = np.logical_and(segs[1], np.logical_not(et))
            ed = np.logical_and(segs[2], np.logical_not(segs[1]))
            labelmap = np.zeros(segs[0].shape)

            labelmap[et] = 3
            labelmap[net] = 2
            labelmap[ed] = 1
            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
            refmap_et = ref_seg == 3
            refmap_tc = np.logical_or(refmap_et, ref_seg == 2)
            refmap_wt = np.logical_or(refmap_tc, ref_seg == 1)
            refmap = np.stack([refmap_et, refmap_tc, refmap_wt])
            patient_metric_list, dice_per_case = calculate_metrics(segs, refmap, patient_id)
            metrics_list.append(patient_metric_list)

            total_metric += dice_per_case

            if save_result:
                sitk.WriteImage(prediction, f"{test_save_path}/{patient_id}.nii.gz")
            ith += 1

    avg_metric = total_metric / len(testloader)
    print('average metric is decoder 1 {}'.format(avg_metric))

    with open(test_save_path + '../{}_performance.txt'.format("vnet"), 'w') as f:
        f.writelines('average metric of decoder 1 is {} \n'.format(avg_metric))

    return avg_metric


def test_all_case_weights(model1, model2, testloader, patch_size=(128, 128, 96), save_result=True, test_save_path=None, w1=0.8, w2=0.2):
    ith = 0
    total_metric = 0.0
    total_metric_average = 0.0
    metrics_list = []
    model1.eval()
    model2.eval()
    with torch.no_grad():
        for step, batch in enumerate(testloader):
            volume_batch, label_batch, seg_path, crops_idx = batch['image'], batch['label'], batch['seg_path'][0], \
                                                             batch['crop_indexes']
            val_inputs, val_labels = volume_batch.cuda(), label_batch.cuda()

            val_inputs = val_inputs.cuda()

            ref_seg_img = sitk.ReadImage(seg_path)
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)

            with torch.no_grad():
                val_outputs_1_1 = inference(val_inputs, model1, patch_size)

                val_outputs_1_2 = inference(val_inputs, model2, patch_size)

                y = torch.zeros(val_outputs_1_1.shape).cuda()
                y = w1 * val_outputs_1_1 + w2 * val_outputs_1_2

            val_outputs = [post_trans(i) for i in decollate_batch(y)]

            segs = torch.zeros((1, 3, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
            segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = val_outputs[0]
            segs = segs[0].numpy() > 0.5

            et = segs[0]
            net = np.logical_and(segs[1], np.logical_not(et))
            ed = np.logical_and(segs[2], np.logical_not(segs[1]))
            labelmap = np.zeros(segs[0].shape)

            labelmap[et] = 3
            labelmap[net] = 2
            labelmap[ed] = 1
            print(np.unique(labelmap))

            labelmap = sitk.GetImageFromArray(labelmap)
            labelmap.CopyInformation(ref_seg_img)
            prediction = labelmap

            patient_id = seg_path.split('/')[-1]
            ref_seg = sitk.GetArrayFromImage(ref_seg_img)
            refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
            refmap_et = ref_seg == 3
            refmap_tc = np.logical_or(refmap_et, ref_seg == 2)
            refmap_wt = np.logical_or(refmap_tc, ref_seg == 1)
            refmap = np.stack([refmap_et, refmap_tc, refmap_wt])
            patient_metric_list, dice_per_case = calculate_metrics(segs, refmap, patient_id)
            metrics_list.append(patient_metric_list)

            total_metric += dice_per_case

            if save_result:
                sitk.WriteImage(prediction, f"{test_save_path}/{patient_id}.nii.gz")
            ith += 1

    avg_metric = total_metric / len(testloader)
    print('average metric is decoder 1 {}'.format(avg_metric))

    with open(test_save_path + '../{}_performance.txt'.format("vnet"), 'w') as f:
        f.writelines('average metric of decoder 1 is {} \n'.format(avg_metric))

    return avg_metric

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


HAUSSDORF = "haussdorf"
DICE = "dice"
IOU = "iou"
ASD = "asd"
METRICS = [DICE, IOU, HAUSSDORF, ASD]