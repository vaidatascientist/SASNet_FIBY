import torch
import torch.nn as nn

class DenLoss(nn.Module):
    def __init__(self):
        super(DenLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, D_preds, D_gts):
        # D_preds and D_gts are lists containing density maps for each of the 5 scales
        loss = 0
        for D_pred, D_gt in zip(D_preds, D_gts):
            loss += self.mse_loss(D_pred, D_gt)
        return loss

class ConfLoss(nn.Module):
    def __init__(self):
        super(ConfLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, D_preds, D_gts):
        # D_preds and C_gts are lists containing confidence maps for each of the 5 scales
        loss = 0
        for C_pred, C_gt in zip(D_preds, D_gts):
            loss += self.bce_loss(C_pred, C_gt)
        return loss

class PRALoss(nn.Module):
    def __init__(self, lambda_):
        super(PRALoss, self).__init__()
        self.lambda_ = lambda_

    def divide_into_subregions(self, region):
        # Assuming region is a square, we split it into 4 equal subregions
        h, w = region.shape[-2], region.shape[-1]
        return region[..., :h//2, :w//2], region[..., :h//2, w//2:], region[..., h//2:, :w//2], region[..., h//2:, w//2:]

    def crowd_count(self, region):
        # Summing up the density values to get the crowd count
        return torch.sum(region)

    def is_over_estimated(self, subregion, gt_subregion):
        # Check if the subregion is over-estimated compared to ground truth
        return self.crowd_count(subregion) > self.crowd_count(gt_subregion)

    def recursive_search(self, region, gt_region):
        if min(region.shape[-2], region.shape[-1]) <= 1:  # If we've reached the smallest possible region
            if self.is_over_estimated(region, gt_region):  # Check if it's over-estimated
                return torch.ones_like(region)
            else:
                return torch.zeros_like(region)

        # Divide the region into sub-regions
        subregions = self.divide_into_subregions(region)
        gt_subregions = self.divide_into_subregions(gt_region)

        hard_pixels = torch.zeros_like(region)
        h_step, w_step = region.shape[-2]//2, region.shape[-1]//2
        for idx, (subregion, gt_subregion) in enumerate(zip(subregions, gt_subregions)):
            y_step = h_step if idx > 1 else 0
            x_step = w_step if idx % 2 else 0
            hard_pixels[..., y_step:y_step + h_step, x_step:x_step + w_step] = \
                self.recursive_search(subregion, gt_subregion)

        return hard_pixels

    def forward(self, D_est, D_gt):
        hard_pixels_mask = self.recursive_search(D_est, D_gt)
        
        G = torch.flatten(D_est)
        H = torch.flatten(D_est * hard_pixels_mask)

        D_est_p = torch.flatten(D_est)
        D_gt_p = torch.flatten(D_gt)
        # exit(0)
        loss = (torch.norm(D_est_p - D_gt_p, p=2)**2 / torch.norm(G, p=2)**2) + \
               self.lambda_ * (torch.norm(D_est_p - D_gt_p, p=2)**2 / torch.norm(H, p=2)**2)

        return loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=10.0):
        super(CombinedLoss, self).__init__()
        self.pra_loss = PRALoss(lambda_= 1)
        self.den_loss = DenLoss()
        self.conf_loss = ConfLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, D_preds, D_gts):
        pra = self.pra_loss(D_preds[0], D_gts[0])  # Assuming the main density map is the first in the list
        den = self.den_loss(D_preds, D_gts)
        conf = self.conf_loss(D_preds, D_gts)
        return self.alpha * pra + self.beta * den + self.gamma * conf
