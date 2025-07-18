import torch
import numpy as np
import cv2
import argparse
import random
import os
import yaml
from tqdm import tqdm
from model import FasterRCNN
from custom_dataset import PerImageAnnotationDataset
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    
    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, score], ...],
    #       'car' : [[x1, y1, x2, y2, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2], ...],
    #       'car' : [[x1, y1, x2, y2], ...]
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]
    
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]
        
        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]
        
        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])
        
        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)
        
        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1
            
            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            
            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]
                
                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    return mean_ap, all_aps


def load_model_and_dataset(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    os.makedirs('data/test/test_image', exist_ok=True)
    os.makedirs('data/test/test_annotation', exist_ok=True)



    voc = PerImageAnnotationDataset(image_dir=dataset_config['im_test_path'], annotation_dir=dataset_config['ann_test_path'])
    test_dataset = DataLoader(voc, batch_size=1, shuffle=False)
    
    faster_rcnn_model = FasterRCNN(model_config, num_classes=dataset_config['num_classes'])
    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(torch.load("model.pth",
                                                 map_location=device))
    return faster_rcnn_model, voc, test_dataset


def infer(args):
    if not os.path.exists('samples'):
        os.mkdir('samples')

    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)
    
    # Hard coding the low score threshold for inference on images for now
    # Should come from config
    faster_rcnn_model.roi_head.low_score_threshold = 0.6
    

    if args.path != 'NULL' and os.path.isfile(args.path):
        # Single image inference from path
        image_path = args.path
        im_np = cv2.imread(image_path)
        if im_np is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        im_rgb = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)
        im_tensor = torch.from_numpy(im_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        im_tensor = im_tensor.to(device)

        rpn_output, frcnn_output = faster_rcnn_model(im_tensor, None)
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        im_copy = im_np.copy()

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label_name = voc.idx2label[labels[idx].detach().cpu().item()]
            score = scores[idx].detach().cpu().item()
            text = f'{label_name} : {score:.2f}'

            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy , (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), (255, 255, 255), -1)
            cv2.putText(im_copy, text, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        output_path = os.path.join('samples', 'output_single_image.jpg')
        cv2.imwrite(output_path, im_copy)
        print(f"Inference done for single image. Saved to {output_path}")
        return output_path
    else:
        for sample_count in tqdm(range(10)):
            random_idx = random.randint(0, len(voc))
            im, target, fname = voc[random_idx]
            im = im.unsqueeze(0).float().to(device)

# Convert to numpy for OpenCV
            im_np = im.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            im_np = np.ascontiguousarray((im_np * 255).astype(np.uint8))

            im_copy = im_np.copy()
        
# Saving images with ground truth boxes
            for idx, box in enumerate(target['bboxes']):
                x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
    
                cv2.rectangle(im_np, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
                cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
                text = voc.idx2label[target['labels'][idx].detach().cpu().item()]
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                text_w, text_h = text_size
                cv2.rectangle(im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
                cv2.putText(im_copy, text=voc.idx2label[target['labels'][idx].detach().cpu().item()],
                        org=(x1+5, y1+15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
                cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            im_out = cv2.addWeighted(im_copy, 0.7, im_np, 0.3, 0)
            cv2.imwrite('samples/output_frcnn_gt_{}.png'.format(sample_count), im_out)
        
        # Getting predictions from trained model
            rpn_output, frcnn_output = faster_rcnn_model(im, None)
            boxes = frcnn_output['boxes']
            labels = frcnn_output['labels']
            scores = frcnn_output['scores']
            im_np = im.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            im_np = np.ascontiguousarray((im_np * 255).astype(np.uint8))

            im_copy = im_np.copy()
        
        # Saving images with predicted boxes
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
                cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
                text = '{} : {:.2f}'.format(voc.idx2label[labels[idx].detach().cpu().item()],
                                        scores[idx].detach().cpu().item())
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                text_w, text_h = text_size
                cv2.rectangle(im_copy , (x1, y1), (x1 + 10+text_w, y1 + 10+text_h), [255, 255, 255], -1)
                cv2.putText(im_copy, text=text,
                        org=(x1+5, y1+15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
                cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            im_out = cv2.addWeighted(im_copy, 0.7, im_np, 0.3, 0)
            cv2.imwrite('samples/output_frcnn_{}.jpg'.format(sample_count), im_out)


def evaluate_map(args):
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(args)
    gts = []
    preds = []
    for im, target, fname in tqdm(test_dataset):
        im_name = fname
        im = im.float().to(device)
        target_boxes = target['bboxes'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        rpn_output, frcnn_output = faster_rcnn_model(im, None)

        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        
        pred_boxes = {}
        gt_boxes = {}
        for label_name in voc.idx2label.values():
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []
        
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, score])
        for idx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            gt_boxes[label_name].append([x1, y1, x2, y2])
        
        gts.append(gt_boxes)
        preds.append(pred_boxes)
   
    mean_ap, all_aps = compute_map(preds, gts, method='interp')
    print('Class Wise Average Precisions')
    for idx in range(len(voc.idx2label)):
        print('AP for class {} = {:.4f}'.format(voc.idx2label[idx], all_aps[voc.idx2label[idx]]))
    print('Mean Average Precision : {:.4f}'.format(mean_ap))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn inference')
    parser.add_argument('--config', dest='config_path',
                        default='config.yaml', type=str)
    parser.add_argument('--evaluate', dest='evaluate',
                        default=False, type=bool)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    parser.add_argument('--path', dest='path',
                        default='NULL', type=str,
                        help='Path to the image directory for inference')

    args = parser.parse_args()
    if args.infer_samples:
        infer(args)
    else:
        print('Not Inferring for samples as `infer_samples` argument is False')
        
    if args.evaluate:
        evaluate_map(args)
    else:
        print('Not Evaluating as `evaluate` argument is False')