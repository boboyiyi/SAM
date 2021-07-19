import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x, label=None, weights=None):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)
        # x_feats的norm2都是1，所以后续的cosine distance就可以直接点积，因为分母为1
        # test = torch.norm(x_feats, dim=1)
        # x_feats shape: [B, 512]
        y_hat_feats = self.extract_feats(y_hat)
        # 把y_feats当成常量
        y_feats = y_feats.detach()
        total_loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])

            if label is None:
                id_logs.append({'diff_target': float(diff_target),
                                'diff_input': float(diff_input),
                                'diff_views': float(diff_views)})
            else:
                id_logs.append({f'diff_target_{label}': float(diff_target),
                                f'diff_input_{label}': float(diff_input),
                                f'diff_views_{label}': float(diff_views)})

            loss = 1 - diff_target
            if weights is not None:
                loss = weights[i] * loss

            # 因为loss乘以了weight，所以sim_improvement和loss并非正负关系
            total_loss += loss
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return total_loss / count, sim_improvement / count, id_logs
