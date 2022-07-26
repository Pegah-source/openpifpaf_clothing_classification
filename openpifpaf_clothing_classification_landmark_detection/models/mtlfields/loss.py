import argparse
import logging

import torch

from ...datasets import headmeta


LOG = logging.getLogger(__name__)


class CategoryLoss(torch.nn.Module):
    """Loss function for category classification.

    Args:
        head_meta (AttributeMeta): Meta information on attribute to predict.
    """

    regression_loss = 'l1'
    focal_gamma = 0.0


    def __init__(self, head_meta: headmeta.ClassMeta):
        super().__init__()
        self.meta = head_meta
        self.field_names = ['{}.{}'.format(head_meta.dataset,
                                           head_meta.name)] # it should be deepfashion.attribute_class (refer to the datamodule in ../datasets/deepfashion/)
        self.previous_loss = None

        LOG.debug('category loss for %s: %d channels',
                  self.meta.attribute,
                  self.meta.n_channels)


    @property
    def loss_function(self):
        if self.meta.n_channels == 1: # this will be used if we want to add the attributes
            return torch.nn.BCEWithLogitsLoss(reduction='none')
        elif self.meta.n_channels > 1:
            loss_module = torch.nn.CrossEntropyLoss(reduction='none')
            '''return lambda x, t: loss_module(
                    x, t.to(torch.float).squeeze(1)).unsqueeze(1)'''
            return lambda x, t: loss_module(
                    x, t.to(torch.long).squeeze(1)).unsqueeze(1)
           
    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CategoryLoss')
       
        group.add_argument('--attribute-focal-gamma',
                           default=cls.focal_gamma, type=float,
                           help='use focal loss with the given gamma')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.focal_gamma = args.attribute_focal_gamma


    def forward(self, *args):
        LOG.debug('loss for %s', self.field_names)

        x, t = args
        loss = self.compute_loss(x, t)

        if (loss is not None) and (not torch.isfinite(loss).item()):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(loss, self.previous_loss))
        self.previous_loss = float(loss.item()) if loss is not None else None

        return [loss]


    def compute_loss(self, x, t):
        if t is None:
            return None

        '''c_x = x.shape[1]
        x = x.reshape(-1, c_x)
        c_t = t.shape[1]
        t = t.reshape(-1, c_t)
        # target is [8, 50] as it should be, but the output is like [3872, 50]
        print('this is the target ', t, ' and shape ', t.shape)
        print('this is the output ', x, ' and shape ', x.shape)

        mask = torch.isnan(t).any(1).bitwise_not_()
        print('and this is the mask ', mask, ' and shape ', mask.shape)
        if not torch.any(mask):
            return None

        x = x[mask, :]
        t = t[mask, :]'''


        c_x = x.shape[1]
        x = x.reshape(-1, c_x)
        c_t = t.shape[0]
        t = t.reshape(c_t, -1)

        mask = torch.isnan(t).any(1).bitwise_not_()
        #print('this is the mask ', mask, '  and its shape ', mask.shape)
        if not torch.any(mask):
            return None

        x = x[mask, :]
        t = t[mask]

        loss = self.loss_function(x, t)
        

        if (self.focal_gamma != 0) and self.meta.is_classification:
            if self.meta.n_channels == 1: # BCE
                focal = torch.sigmoid(x)
                focal = torch.where(t < 0.5, focal, 1. - focal)
            else: # CE
                focal = torch.nn.functional.softmax(x, dim=1)
                focal = 1. - focal.gather(1, t.to(torch.long))
            loss = loss * focal.pow(self.focal_gamma)

        loss = loss.mean()
        return loss
