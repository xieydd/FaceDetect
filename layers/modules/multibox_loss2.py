import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp, decode
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']
 
 
class WingLoss(nn.Module):
    def __init__(self, omega=3, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
 
    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        #print((len(loss1) + len(loss2)))
        #return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
        return (loss1.sum() + loss2.sum())
 
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
 
    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''
 
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum())
        #return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
 
class GIOU(nn.Module):
    def __init__(self,loss_type = "giou"):
        super(GIOU, self).__init__()
        self.variance = [0.1,0.2]
        shape = 300
        self.loss_type = loss_type
        self.scale = torch.Tensor([shape, shape, shape, shape]).cuda()
    def forward(self, preds, priors, targets, pos):
        preds = preds.view(-1, 4)
        targets = targets.view(-1, 4)
        boxes_p = decode(preds, priors, self.variance)
        boxes_t = decode(targets, priors, self.variance)
        boxes_p = boxes_p * self.scale
        boxes_t = boxes_t * self.scale
 
        boxes_p = boxes_p[pos]
        boxes_t = boxes_t[pos]
 
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes_p[:,0], boxes_p[:,1], boxes_p[:,2], boxes_p[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes_t[:,0], boxes_p[:,1], boxes_p[:,2], boxes_p[:,3]
 
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
 
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = (w1 * h1 + 1e-16) + w2 * h2 -inter
        iou = inter / union
 
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
 
 
        if self.loss_type == "giou":
            c_area = cw * ch + 1e-16
            giou =  (iou - (c_area - union) / c_area)
            loss = 1- giou
            return loss.sum()
 
        c2 = cw**2 + ch**2 + 1e-16
        rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) -(b1_y1 + b1_y2)) ** 2 / 4
        if self.loss_type == "diou":
            diou = iou - rho2 / c2
            loss = 1- diou
            return loss.sum()
        if self.loss_type == "ciou":
            v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            with torch.no_grad():
                alpha = v / (1 - iou + v)
            ciou = iou - (rho2 / c2 + v * alpha)
            loss = 1- ciou
 
        return loss.sum()
 
 
 
class MultiBoxLoss2(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
 
    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss2, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]
        self.wingloss = WingLoss()
        #self.adaptivewingloss = AdaptiveWingLoss()
 
    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
 
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
 
        #import pdb; pdb.set_trace()
        loc_data, conf_data, landm_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
 
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros
        num_pos_landm = pos1.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        #import pdb
        #pdb.set_trace()
        landm_p = landm_data[pos_idx1].view(-1, 10)[:,:4]
        landm_t1 = landm_t[pos_idx1].view(-1, 10)[:,:4]
 
        #s1 = torch.ones(1,2)
        #s2 = torch.ones(1,4)*3
        #s = torch.cat([s1,s2],dim=-1).cuda()
 
 
        loss_landm = self.wingloss(landm_p, landm_t1)
        #loss_landm = F.smooth_l1_loss(landm_p, landm_t1, reduction='sum')
 
 
        one = torch.tensor(1).cuda()
        pos_mafa = conf_t == one
        num_pos_landm2 = pos_mafa.long().sum(1, keepdim=True)
        N2 = max(num_pos_landm2.data.sum().float(), 1)
        pos_idx2 = pos_mafa.unsqueeze(pos_mafa.dim()).expand_as(landm_data)
 
        landm_p_mafa = landm_data[pos_idx2].view(-1, 10)[:,4:]
        landm_t1_mafa = landm_t[pos_idx2].view(-1, 10)[:,4:]
 
        s1 = torch.ones(1,2)
        s2 = torch.ones(1,4)*3
        s = torch.cat([s1,s2],dim=-1).cuda()
 
        loss_landm_mafa = self.wingloss(landm_p_mafa*s,landm_t1_mafa*s)
        #loss_landm_mafa = F.smooth_l1_loss(landm_p_mafa*s,landm_t1_mafa*s,reduction='sum')
        #loss_landm = self.wingloss(landm_p*s, landm_t*s)
        #loss_landm = self.adaptivewingloss(landm_p, landm_t)
        pos = conf_t != zeros
        conf_t[pos] = 1
 
        # eye landmark loss
        #pos2 = pos.unsqueeze(pos.dim()).expand_as(landm_data)
        #lm_eye_p = landm_data[pos2].view(-1, 10)[:,:4]
        #lm_eye_t = landm_t[pos2].view(-1, 10)[:,:4]
 
        #loss_landm_eye = F.smooth_l1_loss(lm_eye_p, lm_eye_t, reduction='sum')
 
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
 
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
 
        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
 
        one = torch.tensor(1).cuda()
 
        #import pdb
        #pdb.set_trace()
        tmp = torch.where(targets_weighted==one,torch.tensor(0.1).cuda(),torch.tensor(0.0).cuda())
        conf_p[:,1] = conf_p[:,1] - tmp
 
 
 
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
 
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1
        loss_landm_mafa /= N2
        loss_lm = loss_landm + loss_landm_mafa
        return loss_l, loss_c, loss_lm