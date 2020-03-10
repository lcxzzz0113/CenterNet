import torch
import torch.nn as nn
import numpy as np
# from lib.nms.pth_nms import pth_nms


# def nms(dets, thresh):
#     "Dispatch to either CPU or GPU NMS implementations.\
#     Accept dets as tensor""
#     return pth_nms(dets, thresh)

def nms(dets, thresh):
    x1 = dets[:, 0].cpu().detach().numpy()
    y1 = dets[:, 1].cpu().detach().numpy()
    x2 = dets[:, 2].cpu().detach().numpy()
    y2 = dets[:, 3].cpu().detach().numpy()
    scores = dets[:, 4].cpu().detach().numpy()

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        intersect_x1 = np.maximum(x1[i], x1[order[1:]])
        intersect_y1 = np.maximum(y1[i], y1[order[1:]])
        intersect_x2 = np.minimum(x2[i], x2[order[1:]])
        intersect_y2 = np.minimum(y2[i], y2[order[1:]])

        intersect_w = np.maximum(0.0, intersect_x2 - intersect_x1)
        intersect_h = np.maximum(0.0, intersect_y2 - intersect_y1)
        inter = intersect_h * intersect_w
        IoU = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(IoU <= thresh)[0]
        order = order[inds + 1]

    return keep


def soft_nms(dets, thresh, type='gaussian'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    scores = scores[order]

    keep = []
    while order.size > 0:
        i = order[0]
        dets[i, 4] = scores[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[1:]
        scores = scores[1:]

        if type == 'linear':
            inds = np.where(ovr > thresh)[0]
            scores[inds] *= (1 - ovr[inds])
        else:
            scores *= np.exp(- ovr ** 2 / thresh)

        inds = np.where(scores > 1e-3)[0]
        order = order[inds]
        scores = scores[inds]

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]
    
    return keep


def soft_nms_with_var_voting(dets, confidence, thresh):
    confidence = np.exp(confidence / 2.)
    sigma_t = 0.01
    N = len(dets)
    x1 = dets[:, 0].cpu().detach().numpy().copy()
    y1 = dets[:, 1].cpu().detach().numpy().copy()
    x2 = dets[:, 2].cpu().detach().numpy().copy()
    y2 = dets[:, 3].cpu().detach().numpy().copy()
    scores = dets[:, 4].cpu().detach().numpy().copy()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ious = np.zeros((N, N))
    for i in range(N):
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.maximum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas - inter)
        ious[i, :] = ovr.copy()
    i = 0
    while i < N:
        maxpos = dets[i:N, 4].argmax()
        maxpos += i
        dets[[maxpos, i]] = dets[[i, maxpos]]
        confidence[[maxpos, i]] = confidence[[i, maxpos]]
        ious[[maxpos, i]] = ious[[i, maxpos]]
        ious[:, [maxpos, i]] = ious[:, [i, maxpos]]

        ovr_bbox = np.where((ious[i, :N] > 0.01))[0] + i
        p = np.exp(-(1 - ious[i, ovr_bbox]) ** 2 / sigma_t)
        dets[i, :4] = p.dot(dets[ovr_bbox, :4] / confidence[ovr_bbox] ** 2) / \
            p.dot(1. / confidence[ovr_bbox] ** 2) 
        pos = i + 1
        while pos < N:
            if ious[i, pos] > 0:
                ovr = ious[i, pos]
                dets[pos, 4] *= np.exp(-(ovr * ovr) / thresh)
                if dets[pos, 4] < 0.001:
                    dets[[pos, N - 1]] = dets[[N - 1, pos]]
                    confidence[[pos, N - 1]] = confidence[[N - 1, pos]]
                    ious[[pos, N - 1]] = ious[[N - 1, pos]]
                    ious[:, [pos, N - 1]] = ious[:, [N - 1, pos]]
                    N -= 1
                    pos -= 1
            pos += 1
        i += 1
    keep = [i for i in range(N)]
    return dets[keep]


def box_iou(box1, box2, order='xxyy'):
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou


def box_c(box1, box2, order='xxyy'):
    if order == 'xxyy':
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    xmin = torch.min(b1_x1,b2_x1)
    ymin = torch.min(b1_y1,b2_y1)
    xmax = torch.max(b1_x2,b2_x2)
    ymax = torch.max(b1_y2,b2_y2)

    return (xmin,ymin,xmax,ymax)


def box_union(box1,box2,order='xxyy'):
    if order == 'xxyy':
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    box1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    box2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    intersect = (torch.min(b2_x2,b1_x2) - torch.max(b1_x1,b2_x1)) * (torch.min(b2_y2,b1_y2) - torch.max(b1_y1,b2_y1))
    union_item = (box1_area + box2_area) - intersect + 1e-16

    return union_item


def giou(bbox1,bbox2,order='xxyy'):
    iou_item  = bbox_iou(bbox1, bbox2, order)
    boxc_item = box_c(bbox1, bbox2, order)
    boxc_area = (boxc_item[2] - boxc_item[0]) * (boxc_item[3] - boxc_item[1]) +  1e-7

    if boxc_area.eq(0).sum() > 0 :
        return iou_item
    u = box_union(bbox1, bbox2, order)
    giou_item = iou_item - (boxc_area - u) / boxc_area
    return giou_item


def change_box_order(boxes, order):
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a+1], 1)
    return torch.cat([a-b/2,a+b/2], 1)


def meshgrid(x, y, row_major=True):
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)


def box_nms(bboxes, scores, threshold=0.9, mode='union'):
    x1 = bboxes[:, 0].cpu().detach().numpy()
    y1 = bboxes[:, 1].cpu().detach().numpy()
    x2 = bboxes[:, 2].cpu().detach().numpy()
    y2 = bboxes[:, 3].cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        intersect_x1 = np.maximum(x1[i], x1[order[1:]])
        intersect_y1 = np.maximum(y1[i], y1[order[1:]])
        intersect_x2 = np.minimum(x2[i], x2[order[1:]])
        intersect_y2 = np.minimum(y2[i], y2[order[1:]])

        intersect_w = np.maximum(0.0, intersect_x2 - intersect_x1)
        intersect_h = np.maximum(0.0, intersect_y2 - intersect_y1)
        inter = intersect_h * intersect_w
        IoU = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(IoU <= threshold)[0]
        order = order[inds + 1]

    return torch.LongTensor(keep)


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
