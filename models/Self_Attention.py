import torch
import torch.nn as nn

class Self_Attention(nn.Module):

    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()

        #1x1 pointwise convolution
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels= in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels= in_dim, out_channels=in_dim, kernel_size=1)


        self.softmax = nn.Softmax(dim=-2)

        #out = x + gamma * o
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        X = x

        #B, C', W, H -> B, C', N  , where N is WxH
        proj_query = self.query_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])
        #transpose
        proj_query = proj_query.permute(0, 2, 1)
        #B, C', W, H -> B, C', N  , where N is WxH
        proj_key = self.key_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])

        #batchごとのmatmul
        S = torch.bmm(proj_query, proj_key)

        #self.softmax = nn.Softmax(dim=-2), normalize(Row-wise)
        attention_map_T = self.softmax(S)
        #transpose
        attention_map = attention_map_T.permute(0, 2, 1)

        proj_value = self.value_conv(X).view(X.shape[0], -1, X.shape[2]*X.shape[3])
        # matmul after transposing attention_map
        o = torch.bmm(proj_value, attention_map.permute(0, 2, 1))

        o = o.view(X.shape[0], X.shape[1], X.shape[2], X.shape[3])

        out = x+self.gamma*o

        return out, attention_map