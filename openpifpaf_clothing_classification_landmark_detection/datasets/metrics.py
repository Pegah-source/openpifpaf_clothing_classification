import logging
import json
import zipfile

import numpy as np
import openpifpaf

try:
    import pycocotools.coco
    from pycocotools.cocoeval import COCOeval
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    COCOeval = None

LOG = logging.getLogger(__name__)

import torch

class LandmarkEvaluator(object):

    def __init__(self):

        self.reset()

    def reset(self):
        # QUESTION: how to compute the last two??
        self.lm_vis_count_all = np.array([0.] * 8)
        self.lm_inpic_count_all = np.array([0.] * 8)
        self.lm_dist_vis = np.array([0.] * 8)
        self.lm_dist_inpic = np.array([0.] * 8)
        self.vis_n_pred = np.array([0.] * 8)
        self.inpic_n_pred = np.array([0.] * 8)

    def landmark_count(self, pred_landmarks, gt_landmarks, w, h):

        # creating visible and in_pic masks
        visibilities = [gt_landmarks[3*i+2] for i in range(8)].reshape((-1, 8, 1))
        visible_mask = np.where(visibilities==2 , 1, 0).reshape((-1, 8, 1))
        visible_mask = torch.from_numpy(visible_mask)
        inpic_mask = np.where(visibilities>=1 , 1, 0).reshape((-1, 8, 1))
        inpic_mask = torch.from_numpy(inpic_mask)


        # rearranging the landmarks
        pred_landmarks_rearrange = [pred_landmarks[3*i+j] for i in range(8) for j in range(2)].reshape((-1, 8, 2))
        gt_landmarks_rearrange = [gt_landmarks[3*i+j] for i in range(8) for j in range(2)].reshape((-1, 8, 2))


        # normalizing the position of landmarks gt and in pred
        a = [float(w), float(h)]
        a = np.expand_dims(a, axis = 1)
        b = np.concatenate([a.T for i in range(8)], axis = 0)
        c = np.stack([b], axis = 0)

        pred_landmarks_rearrange = pred_landmarks_rearrange / c
        gt_landmarks_rearrange = gt_landmarks_rearrange / c

        # counting the visible and in_pic landmarks
        landmark_vis_count = visible_mask.cpu().numpy().sum(axis=0)
        landmark_inpic_count = inpic_mask.cpu().numpy().sum(axis=0)


        # creating visibility and in_pic arrays to multiply with the difference of landmarks
        landmark_vis_float = torch.unsqueeze(visible_mask.float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2).cpu().detach().numpy()

        landmark_inpic_float = torch.unsqueeze(inpic_mask.float(), dim=2)
        landmark_inpic_float = torch.cat([landmark_inpic_float, landmark_inpic_float], dim=2).cpu().detach().numpy()



        landmark_dist_vis = np.sum(np.sqrt(np.sum(np.square(
            landmark_vis_float * pred_landmarks_rearrange - landmark_vis_float * gt_landmarks_rearrange,
        ), axis=2)), axis=0)

        landmark_dist_inpic = np.sum(np.sqrt(np.sum(np.square(
            landmark_inpic_float * pred_landmarks_rearrange - landmark_inpic_float * gt_landmarks_rearrange,
        ), axis=2)), axis=0)

        self.lm_vis_count_all += landmark_vis_count
        self.lm_inpic_count_all += landmark_inpic_count


        self.lm_dist_vis += landmark_dist_vis
        self.lm_dist_inpic += landmark_dist_inpic



    def add(self, pred_landmarks, gt_landmarks, w, h):
        self.landmark_count(pred_landmarks, gt_landmarks, w, h)

    def evaluate(self):
        lm_individual_dist_vis = self.lm_dist_vis / self.lm_vis_count_all
        lm_individual_dist_inpic = self.lm_dist_inpic / self.lm_inpic_count_all

        lm_dist_vis = (self.lm_dist_vis / self.lm_vis_count_all).mean()
        lm_dist_inpic = (self.lm_dist_inpic / self.lm_inpic_count_all).mean()

        return {
            'lm_individual_dist_vis': lm_individual_dist_vis,
            'lm_dist_vis': lm_dist_vis,
            'lm_individual_dist_inpic' : lm_individual_dist_inpic,
            'lm_dist_inpic' : lm_dist_inpic
        }



'''
    LANDMARK EVALUATOR FOR DEEPFASHION DATASET:

        1. NORMALIZED DISTANCE OF PREDICTED AND GROUND-TRUTH KEYPOINTS OVER THE VISIBLE KEYPOINTS
        2. NORMALIZED DISTANCE OF PREDICTED AND GROUND-TRUTH KEYPOINTS OVER THE KEYPOINTS IN THE PICTURE
'''

class Deepfashion_landmark(openpifpaf.metric.base.Base):
    

    def __init__(self, coco_anno_file, confidence_threshold=0.0):

        super().__init__()

        self.coco_anno_file = coco_anno_file
        self.coco_eval = pycocotools.coco.COCO(coco_anno_file)
        self.small_threshold = confidence_threshold

        self.predictions = {}
        self.indices = []


    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        # Store predictions for writing to file


        # the predictions are of type combined_annotation
        # the function should be used to fill predictions dictionary initialized in __init__

        pred_data = []
        for pred in predictions:
            pred_data.append(pred.json_data())
        self.predictions[image_meta['image_id']] = pred_data
        self.indices.append(image_meta['image_id'])



    def stats(self):

        if predictions is None:
            predictions = self.predictions
        if image_ids is None:
            image_ids = self.image_ids


        evaluator = LandmarkEvaluator()


        for index in self.indices:
            # keypoints of the gt:
            anns = self.coco_eval.loadAnns(index)
            img = self.coco_eval.loadImgs(index)[0]
            w, h = int(img['width']), int(img['height'])
            keypoints_gt = anns[0]['keypoints']

            # keypoints of prediction
            pred = predictions[index]
            pred_data = pred.json_data()
            keypoints_pred = pred_data['keypoints']


            # adding two sets of keypoints to the evaluator
            evaluator.add(keypoints_pred, keypoints_gt, w, h)


        result_dict = evaluator.evaluate()
        output_dict = {}
        output_dict['text_labels'] = result_dict.keys()
        output_dict['stats'] = result_dict.values()


        return output_dict

    def write_predictions(self, filename, *, additional_data=None):
        predictions = [
            {k: v for k, v in annotation.items()
             if k in ('image_id', 'category_id', 'keypoints', 'score')}
            for annotation in self.predictions
        ]
        with open(filename + '.pred.json', 'w') as f:
            json.dump(predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        if additional_data:
            with open(filename + '.pred_meta.json', 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)








class ClassificationEvaluator(object):

    def __init__(self, category_topk=(1, 3, 5)):
        self.category_topk = category_topk
        self.reset()

    def reset(self):

        self.category_accuracy = []


    def category_topk_accuracy(self, output, target):
        # what should the output be like??
        
        with torch.no_grad():
            maxk = max(self.category_topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in self.category_topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100 / batch_size))
            for i in range(len(res)):
                res[i] = res[i].cpu().numpy()[0] / 100

            self.category_accuracy.append(res)        

    def evaluate(self):
        category_accuracy = np.array(self.category_accuracy).mean(axis=0)
        category_accuracy_topk = {}
        for i, top_n in enumerate(self.category_topk):
            category_accuracy_topk[top_n] = category_accuracy[i]

       
        return {
            'category_accuracy_topk': category_accuracy_topk[0],
            'category_accuracy_top3' : category_accuracy_topk[1],
            'category_accuracy_top5' : category_accuracy_topk[2],
        }


    
'''
    CLASSIFICATION EVALUATOR FOR DEEPFASHION DATASET:

        1. topk accuracies
'''

class Deepfashion_classification(openpifpaf.metric.base.Base):
    

    def __init__(self, coco_anno_file):
        super().__init__()

        self.coco_anno_file = coco_anno_file
        self.coco_eval = pycocotools.coco.COCO(coco_anno_file)

        self.predictions = {}
        self.indices = []

    def accumulate(self, predictions, image_meta, *, ground_truth=None):
        # category_id in annotations
        pred_data = []
        for pred in predictions:
            pred_data.append(pred.json_data()[''])
        self.predictions[image_meta['image_id']] = pred_data
        self.indices.append(image_meta['image_id'])



    def stats(self):

        if predictions is None:
            predictions = self.predictions
        if image_ids is None:
            image_ids = self.image_ids


        evaluator = ClassificationEvaluator()


        for index in self.indices:
            # keypoints of the gt:
            anns = self.coco_eval.loadAnns(index)
            cat_gt = anns[0]['category_id'] # this should be the index of the top category

            # keypoints of prediction
            pred = predictions[index]
            pred_data = pred.json_data()
            cat_pred = pred_data['class'] # this should be an array of probabilities


            # adding two sets of keypoints to the evaluator
            evaluator.add(cat_pred, cat_gt)


        result_dict = evaluator.evaluate()
        output_dict = {}
        output_dict['text_labels'] = result_dict.keys()
        output_dict['stats'] = result_dict.values()


        return output_dict

    def write_predictions(self, filename, *, additional_data=None):
        predictions = [
            {k: v for k, v in annotation.items()
             if k in ('image_id', 'category_id', 'keypoints', 'score')}
            for annotation in self.predictions
        ]
        with open(filename + '.pred.json', 'w') as f:
            json.dump(predictions, f)
        LOG.info('wrote %s.pred.json', filename)
        with zipfile.ZipFile(filename + '.zip', 'w') as myzip:
            myzip.write(filename + '.pred.json', arcname='predictions.json')
        LOG.info('wrote %s.zip', filename)

        if additional_data:
            with open(filename + '.pred_meta.json', 'w') as f:
                json.dump(additional_data, f)
            LOG.info('wrote %s.pred_meta.json', filename)


