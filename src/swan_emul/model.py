import torch, torch.nn as nn, torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, c, red=16):
        super().__init__()
        r = max(c // red, 1)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(c, r), nn.ReLU(True),
                                nn.Linear(r, c), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(x)
        return x * w.view(x.size(0), x.size(1), 1, 1)

class ImprovedConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, pad=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, k, padding=pad, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.gn = nn.GroupNorm(32 if out_c % 32 == 0 else 16, out_c)
        self.se = SEBlock(out_c)
        self.act = nn.ReLU(True)
    def forward(self, x):
        x = self.pw(self.dw(x))
        return self.act(self.se(self.gn(x)))

class UNetPlusPlus(nn.Module):
    def __init__(self, in_c, out_c, feat=(32,64,128,256,512)):
        super().__init__()
        f = feat; P = nn.MaxPool2d(2,2)
        self.enc00 = ImprovedConvBlock(in_c, f[0]); self.P = P
        self.enc10 = ImprovedConvBlock(f[0], f[1])
        self.enc20 = ImprovedConvBlock(f[1], f[2])
        self.enc30 = ImprovedConvBlock(f[2], f[3])
        self.enc40 = ImprovedConvBlock(f[3], f[4])
        self.dec01 = ImprovedConvBlock(f[0]+f[1], f[0])
        self.dec11 = ImprovedConvBlock(f[1]+f[2], f[1])
        self.dec21 = ImprovedConvBlock(f[2]+f[3], f[2])
        self.dec31 = ImprovedConvBlock(f[3]+f[4], f[3])
        self.dec02 = ImprovedConvBlock(f[0]*2+f[1], f[0])
        self.dec12 = ImprovedConvBlock(f[1]*2+f[2], f[1])
        self.dec22 = ImprovedConvBlock(f[2]*2+f[3], f[2])
        self.dec03 = ImprovedConvBlock(f[0]*3+f[1], f[0])
        self.dec13 = ImprovedConvBlock(f[1]*3+f[2], f[1])
        self.dec04 = ImprovedConvBlock(f[0]*4+f[1], f[0])
        self.outs = nn.ModuleList([nn.Conv2d(f[0], out_c, 1) for _ in range(4)])
    def _u(self, x, y):
        return torch.cat([F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=False), y], 1)
    def forward(self, x):
        x00 = self.enc00(x)
        x10 = self.enc10(self.P(x00)); x20 = self.enc20(self.P(x10))
        x30 = self.enc30(self.P(x20)); x40 = self.enc40(self.P(x30))
        x01 = self.dec01(self._u(x10, x00))
        x11 = self.dec11(self._u(x20, x10)); x21 = self.dec21(self._u(x30, x20)); x31 = self.dec31(self._u(x40, x30))
        x02 = self.dec02(self._u(x11, torch.cat([x00, x01], 1)))
        x12 = self.dec12(self._u(x21, torch.cat([x10, x11], 1)))
        x22 = self.dec22(self._u(x31, torch.cat([x20, x21], 1)))
        x03 = self.dec03(self._u(x12, torch.cat([x00, x01, x02], 1)))
        x13 = self.dec13(self._u(x22, torch.cat([x10, x11, x12], 1)))
        x04 = self.dec04(self._u(x13, torch.cat([x00, x01, x02, x03], 1)))
        return [self.outs[0](x04), self.outs[1](x03), self.outs[2](x02), self.outs[3](x01)]

class ConvLSTMCell(nn.Module):
    def __init__(self, in_c, hid_c, k=3):
        super().__init__()
        pad = k // 2
        self.h = hid_c
        self.conv = nn.Conv2d(in_c + hid_c, 4 * hid_c, k, padding=pad)
    def forward(self, x, s):
        h, c = s
        i, f, o, g = torch.split(self.conv(torch.cat([x, h], 1)), self.h, 1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g; h = o * torch.tanh(c)
        return h, c
    def init_state(self, B, H, W, dev):
        z = torch.zeros(B, self.h, H, W, device=dev)
        return z.clone(), z.clone()

class UNetConvLSTM(nn.Module):
    def __init__(self, input_channels=6, output_channels=4, hidden_dim=128, feat=(32,64,128,256,512)):
        super().__init__()
        self.unet = UNetPlusPlus(input_channels, output_channels, feat)
        self.lstm = ConvLSTMCell(output_channels, hidden_dim)
        self.head = nn.Conv2d(hidden_dim, output_channels, 1)
    def forward(self, x):
        B, T, C, H, W = x.shape
        h, c = self.lstm.init_state(B, H, W, x.device)
        last_u = None
        for t in range(T):
            outs = self.unet(x[:, t]); last_u = outs
            h, c = self.lstm(outs[0], (h, c))
        return self.head(h)

def load_model(ckpt_path, in_ch=6, out_ch=4, hidden=128, feat=(32,64,128,256,512), maploc="cpu"):
    m = UNetConvLSTM(in_ch, out_ch, hidden, feat)
    obj = torch.load(ckpt_path, map_location=maploc)
    if isinstance(obj, dict) and "state_dict" in obj:
        m.load_state_dict(obj["state_dict"], strict=False)
    elif isinstance(obj, dict):
        m.load_state_dict(obj, strict=False)
    else:
        m = obj
    m.eval()
    return m
