"""
Code taken from: Combining Implicit Function Learning and Parametric Models for 3D Human Reconstruction, ECCV'20
"""

import torch
from collections import deque, defaultdict


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class th_Mahalanobis(object):
    def __init__(self, mean, prec, prefix):
        self.mean = torch.tensor(mean.astype('float32'), requires_grad=False).unsqueeze(axis=0).cuda()
        self.prec = torch.tensor(prec.astype('float32'), requires_grad=False).cuda()
        self.prefix = prefix

    def __call__(self, pose, prior_weight=1.):
        '''
        :param pose: Batch x pose_dims
        :return:
        '''
        # return (pose[:, self.prefix:] - self.mean)*self.prec
        temp = pose[:, self.prefix:] - self.mean
        temp2 = torch.matmul(temp, self.prec) * prior_weight
        return (temp2 * temp2).sum(dim=1)


class Prior(object):
    def __init__(self, sm, prefix=3):
        self.prefix = prefix
        if sm is not None:
            # Compute mean and variance based on the provided poses
            self.pose_subjects = sm.pose_subjects
            all_samples = [p[prefix:] for qsub in self.pose_subjects
                           for name, p in zip(qsub['pose_fnames'], qsub[
                    'pose_parms'])]  # if 'CAESAR' in name or 'Tpose' in name or 'ReachUp' in name]
            self.priors = {'Generic': self.create_prior_from_samples(all_samples)}
        else:
            import pickle as pkl
            # Load pre-computed mean and variance
            dat = pkl.load(open('assets/pose_prior.pkl', 'rb'))
            self.priors = {'Generic': th_Mahalanobis(dat['mean'],
                                                     dat['precision'],
                                                     self.prefix)}

    def create_prior_from_samples(self, samples):
        from sklearn.covariance import GraphicalLassoCV
        from numpy import asarray, linalg
        model = GraphicalLassoCV()
        model.fit(asarray(samples))
        return th_Mahalanobis(asarray(samples).mean(axis=0),
                              linalg.cholesky(model.precision_),
                              self.prefix)

    def __getitem__(self, pid):
        if pid not in self.priors:
            samples = [p[self.prefix:] for qsub in self.pose_subjects
                       for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])
                       if pid in name.lower()]
            self.priors[pid] = self.priors['Generic'] if len(samples) < 3 \
                else self.create_prior_from_samples(samples)

        return self.priors[pid]



def get_loss_weights():
    """Set loss weights"""
    loss_weight = {'chamfer': lambda cst, it: 10. ** 0 * cst * (1 + it),
                   'p2s': lambda cst, it: 10. ** 0 * cst * (1 + it),
                   'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                   'pose_pr': lambda cst, it: 10. ** -1 * cst / (1 + it),
                   }
    return loss_weight


def backward_step(loss_dict, weight_dict, it):
    w_loss = dict()
    for k in loss_dict:
        w_loss[k] = weight_dict[k](loss_dict[k], it)

    tot_loss = list(w_loss.values())
    tot_loss = torch.stack(tot_loss).sum()
    return tot_loss