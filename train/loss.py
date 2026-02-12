import torch
from torch import nn
from torch.autograd import Variable


class IOULoss(nn.Module):
    def __init__(self, size=224, activate=nn.Sigmoid()):
        super(IOULoss, self).__init__()
        self.size = size
        self.activate = activate

    def forward(self, pred, target, weight=None):
        pred = self.activate(pred)
        pred_left, pred_top, pred_right, pred_bottom = (pred[:,0]-pred[:,2])*self.size, (pred[:,1]-pred[:,3])*self.size, (pred[:,0]+pred[:,2])*self.size, (pred[:,1]+pred[:,3])*self.size
        target_left, target_top, target_right, target_bottom = (target[:,0]-target[:,2])*self.size, (target[:,1]-target[:,3])*self.size, (target[:,0]+target[:,2])*self.size, (target[:,1]+target[:,3])*self.size

        union_area = (target_right-target_left)*(target_bottom-target_top) + (pred_right-pred_left)*(pred_bottom-pred_top)

        length = torch.min(pred_right,target_right)-torch.max(pred_left,target_left)
        wide = torch.min(pred_bottom,target_bottom)-torch.max(pred_top,target_top)
        length[length>=0.] += 1.
        wide[wide>=0.] += 1.
        length[length<0.] = 0.
        wide[wide<0.] = 0.

        interset_area = (length)*(wide)
        union_area = union_area-interset_area

        ious = (interset_area+1.)/(union_area+1.)

        losses = 1 - ious

        if weight is not None:
            return losses * weight
        else:
            return losses

class Dice(nn.Module):
    def __init__(self, size=224, activate=nn.Sigmoid()):
        super(Dice, self).__init__()
        self.size = size
        self.activate = activate

    def forward(self, pred, target, weight=None):
        pred = self.activate(pred)
        pred_left, pred_top, pred_right, pred_bottom = (pred[:,0]-pred[:,2])*self.size, (pred[:,1]-pred[:,3])*self.size, (pred[:,0]+pred[:,2])*self.size, (pred[:,1]+pred[:,3])*self.size
        target_left, target_top, target_right, target_bottom = (target[:,0]-target[:,2])*self.size, (target[:,1]-target[:,3])*self.size, (target[:,0]+target[:,2])*self.size, (target[:,1]+target[:,3])*self.size

        # print(pred_left)
        # pred_left, pred_top, pred_right, pred_bottom = pred[:,0]*self.size, pred[:,1]*self.size, pred[:,2]*self.size, pred[:,3]*self.size
        # target_left, target_top, target_right, target_bottom = target[:,0]*self.size, target[:,1]*self.size, target[:,2]*self.size, target[:,3]*self.size

        union_area = (target_right-target_left)*(target_bottom-target_top) + (pred_right-pred_left)*(pred_bottom-pred_top)
        length = torch.min(pred_right,target_right)-torch.max(pred_left,target_left)
        wide = torch.min(pred_bottom,target_bottom)-torch.max(pred_top,target_top)
        length[length>=0.] += 1.
        wide[wide>=0.] += 1.
        length[length<0.] = 0.
        wide[wide<0.] = 0.

        interset_area = (length)*(wide)

        losses =1.- (2*interset_area+1.)/(union_area+1.)
        if weight is not None:
            union_area = union_area-2*interset_area
            union_area = union_area - union_area.min()
            losses = losses * nn.Softmax()(union_area / union_area.max())
            return losses.sum()
        else:
            return losses
        
class FocalLoss(nn.Module):
    def __init__(self, size, class_num=4, alpha=None, gamma=2, size_average=True, activate=nn.Sigmoid()):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.activate = activate

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = self.activate(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss