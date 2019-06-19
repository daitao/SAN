
import torch
import torch.nn as nn
import torch.nn.functional as F


class gradient_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(gradient_loss, self).__init__()

        sobel_x = torch.Tensor([[1,0,-1],[2,0,-2],[1,0,-1]])
        sobel_y = torch.Tensor([[1,2,1],[0,0,0],[-1,-2,-1]])
        sobel_3x = torch.Tensor(1,3,3,3)
        sobel_3y = torch.Tensor(1,3,3,3)
        sobel_3x[:,0:3,:,:] = sobel_x
        sobel_3y[:,0:3,:,:] = sobel_y
        self.conv_hx = nn.Conv2d(3,3, kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_hy = nn.Conv2d(3,3, kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_hx.weight = torch.nn.Parameter(sobel_3x)
        self.conv_hy.weight = torch.nn.Parameter(sobel_3y)

        # self.L1_loss = torch.nn.L1Loss()

    def forward(self, X, Y):
        # compute gradient of X
        # batch_size,C,H,W = X.shape
        # X_r = X[:, :1, :H, :W]
        # Y_r = Y[:, :1, :H, :W]
        X_hx = self.conv_hx(X)
        X_hy = self.conv_hy(Y)
        G_X = torch.abs(X_hx) + torch.abs(X_hy)
        # compute gradient of Y
        Y_hx = self.conv_hx(Y)
        self.conv_hx.train(False)
        Y_hy = self.conv_hy(Y)
        self.conv_hy.train(False)
        G_Y = torch.abs(Y_hx) + torch.abs(Y_hy)

        # loss = self.L1_loss(G_X,G_Y)
        loss = F.mse_loss(G_X, G_Y,size_average=True)

        return loss