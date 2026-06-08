import math
import numpy as np
from functools import partial
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass
from model.modules.HWD import HWD
from model.modules.STDFG import STDFG
from model.modules.SDFG import SDFG
from model.modules.DWConv import DWConv
from model.modules.NED import NED

class DWConv_BN_GELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.dwconv = DWConv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, input):
        return self.gelu(self.bn(self.dwconv(input)))

def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    assert not with_complex
    flops = 0
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
 

    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
   
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    
    return flops

class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H//2, W//2, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1) 
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) 
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class CSTFE_block(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ln_before = norm_layer(hidden_dim)
        self.self_attention_before = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path_before = DropPath(drop_path)

        self.ln_after = norm_layer(hidden_dim*2)
        self.self_attention_after = SS2D(d_model=hidden_dim*2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path_after = DropPath(drop_path)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        x1 = input1 + self.drop_path_before(self.self_attention_before(self.ln_before(input1)))
        x2 = input2 + self.drop_path_before(self.self_attention_before(self.ln_before(input2)))

        cross_feat = self.drop_path_after(self.self_attention_after(self.ln_after(torch.cat([x1,x2],dim=3))))

        out1 = input1 + torch.sigmoid(cross_feat[:,:,:,:self.hidden_dim]) * input1
        out2 = input2 + torch.sigmoid(cross_feat[:,:,:,self.hidden_dim:]) * input2
        return out1, out2

class CSTFE(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            CSTFE_block(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])
        
        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x1, x2):
        for blk in self.blocks:
            if self.use_checkpoint:
                x1, x2 = checkpoint.checkpoint(blk, x1, x2)
            else:
                x1, x2 = blk(x1, x2)
        
        if self.downsample is not None:
            x1 = self.downsample(x1)
            x2 = self.downsample(x2)
    
        x1 = x1.permute(0,3,1,2)
        x2 = x2.permute(0,3,1,2)

        return x1, x2

class STNet(nn.Module):
    def __init__(self, in_chans, num_classes, vss_depth, vssm_path_size) -> None:
        super().__init__()
        self.SIE1 = nn.Sequential(DWConv_BN_GELU(in_channels=in_chans, out_channels=32,kernel_size=3,stride=1,padding=1),
                                    DWConv_BN_GELU(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),
                                    HWD(in_channels=32,out_channels=64),
        )
        self.SIE2 = nn.Sequential(DWConv_BN_GELU(in_channels=64, out_channels=96,kernel_size=3,stride=1,padding=1),
                                    DWConv_BN_GELU(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding=1),
                                    HWD(in_channels=96,out_channels=128),
        )
        self.SIE3 = nn.Sequential(DWConv_BN_GELU(in_channels=128, out_channels=192,kernel_size=3,stride=1,padding=1),
                                    DWConv_BN_GELU(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1),
                                    DWConv_BN_GELU(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1),
                                    HWD(in_channels=192,out_channels=256),
        )
        self.patch_embed = PatchEmbed2D(patch_size=vssm_path_size, in_chans=256, embed_dim=256,
            norm_layer=nn.LayerNorm)
        self.CSTFE1 = CSTFE(dim=256,depth=vss_depth,downsample=None)
        self.CSTFE2 = CSTFE(dim=256,depth=vss_depth,downsample=PatchMerging2D)
        self.STDFG1 = STDFG(in_channels=512, spatial_channels=256,out_channels=512)
        self.STDFG2 = STDFG(in_channels=256, spatial_channels=128,out_channels=256)
        self.SDFG3 = SDFG(in_channels=256)
        self.SDFG2 = SDFG(in_channels=128)
        self.SDFG1 = SDFG(in_channels=64)
        self.ned = NED(in_channels_list=[
                                [64+128, 64+128+256, 128+256+256, 256+256+512],
                                [64+128, 64+128+256, 128+256+256],
                                [64+128, 64+128+256],
                                [64+128]
                            ],
                            out_channel_list=[
                                [64, 128, 256, 256],
                                [64, 128, 256],
                                [64, 128],
                                [64]
                            ],
                            aux=True
                            )
        self.decoder = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
        )
        self.head = nn.Conv2d(in_channels=16, out_channels=num_classes, kernel_size=1)
        if self.training:
            self.conv1x1_5 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
            self.conv1x1_4 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
            self.conv1x1_3 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
            self.conv1x1_2 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
            self.conv1x1_1 = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    
    def forward(self, input1,input2):
    
        sie1_A = self.SIE1(input1)
        sie1_B = self.SIE1(input2)  
        sie2_A = self.SIE2(sie1_A)
        sie2_B = self.SIE2(sie1_B)  
        sie3_A = self.SIE3(sie2_A)
        sie3_B = self.SIE3(sie2_B)  

        patch_embed_A = self.patch_embed(sie3_A)
        patch_embed_B = self.patch_embed(sie3_B)
        cstfe1_A, cstfe1_B = self.CSTFE1(patch_embed_A, patch_embed_B)
        cstfe2_A, cstfe2_B = self.CSTFE2(cstfe1_A.permute(0,2,3,1), cstfe1_B.permute(0,2,3,1))

        stdfg1 = self.STDFG1(cstfe2_A, cstfe2_B)
        stdfg2 = self.STDFG2(cstfe1_A, cstfe1_B)
        sdfg1 = self.SDFG1(sie1_A, sie1_B)
        sdfg2 = self.SDFG2(sie2_A, sie2_B)
        sdfg3 = self.SDFG3(sie3_A, sie3_B)

        output1, output2, output3, output4, output5 = self.ned(input1=sdfg1, input2=sdfg2, input3=sdfg3, input4=stdfg2, input5=stdfg1)
        output = self.decoder(F.interpolate(output5, size=input1.size()[2:],mode='bilinear', align_corners=True))
        output = self.head(output)
        if self.training:
            return output, \
                    F.interpolate(self.conv1x1_5(output5), size=input1.size()[2:], mode='bilinear', align_corners=True),\
                    F.interpolate(self.conv1x1_4(output4), size=input1.size()[2:], mode='bilinear', align_corners=True),\
                    F.interpolate(self.conv1x1_3(output3), size=input1.size()[2:], mode='bilinear', align_corners=True),\
                    F.interpolate(self.conv1x1_2(output2), size=input1.size()[2:], mode='bilinear', align_corners=True),\
                    F.interpolate(self.conv1x1_1(output1), size=input1.size()[2:], mode='bilinear', align_corners=True)
        return output
