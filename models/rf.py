import joblib

import torch
from .base_models import BaseIconModel


class RandomForest(BaseIconModel):
    def __init__(self, x3d_mean, x3d_std, x2d_mean, x2d_std, args):
        self.model = joblib.load(args.rf_model_path)
        self.feats_out = len(args.feats_out)
        self.height_out = args.height_out 
        self.scale_output = args.scale_output
        super().__init__(x3d_mean, x3d_std, x2d_mean, x2d_std)

    
    def __call__(self, x3d, x2d):
        x2d_org = x2d.clone()
        x = torch.cat([x3d.flatten(1, 2), x2d], dim=1)
        y = self.model.predict(x).reshape(-1, self.height_out, self.feats_out)
        return self._scale_output(torch.from_numpy(y), x2d_org)
        