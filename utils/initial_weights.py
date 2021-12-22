import torch
import torch.nn as nn

def initialize_weights(self) -> None:
    for m in self.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Conv1d)): # 30 36
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # 26 29
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)