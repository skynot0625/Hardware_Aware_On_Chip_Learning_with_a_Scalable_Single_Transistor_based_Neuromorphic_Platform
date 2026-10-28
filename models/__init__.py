"""Select original research classes without wrapping or rewriting their forward methods."""

from torch import nn

from . import model_Arch1, model_Arch2
from . import model_Resnet18_SG, model_Resnet_cSG, model_Resnet_SG_eLIFE, model_Resenet_cSG_eLIFE


def PreActResNet18(scale: int = 64, num_steps: int = 5, elife: bool = False,
                 surrogate: str = "csg") -> nn.Module:
    classes = {
        ("sg", False): model_Resnet18_SG.SOLO_PreactResNet18,
        ("csg", False): model_Resnet_cSG.SOLO_PreactResNet18,
        ("sg", True): model_Resnet_SG_eLIFE.SOLO_PreactResNet18,
        ("csg", True): model_Resenet_cSG_eLIFE.SOLO_PreactResNet18,
    }
    if (surrogate, elife) not in classes or scale < 1 or num_steps < 1:
        raise ValueError("require sg/csg, positive scale and positive num_steps")
    return classes[surrogate, elife](scale=scale, num_steps=num_steps,
                                    init_LIF_tau=1.0, init_LI_tau=3.0, subthresh=1.0)


def SHDELife(scale: int = 1024, num_steps: int = 20, min_steps: int = 4,
             surrogate: str = "csg", elife: bool = True, architecture: str = "arch2") -> nn.Module:
    if scale != 1024 or num_steps != 20 or not 2 <= min_steps <= 20:
        raise ValueError("original SHD classes require scale=1024, num_steps=20 and 2<=min_steps<=20")
    classes = {
        ("arch1", "sg", False): model_Arch1.mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3_x_pure,
        ("arch1", "csg", False): model_Arch1.mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3_x_pure,
        ("arch1", "sg", True): model_Arch1.mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3_x,
        ("arch1", "csg", True): model_Arch1.mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3_x,
        ("arch2", "sg", False): model_Arch2.mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3_pure,
        ("arch2", "csg", False): model_Arch2.mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3_pure,
        ("arch2", "sg", True): model_Arch2.mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3,
        ("arch2", "csg", True): model_Arch2.mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3,
    }
    if (architecture, surrogate, elife) not in classes:
        raise ValueError("require architecture arch1/arch2 and surrogate sg/csg")
    return classes[architecture, surrogate, elife](scale=scale, num_steps=num_steps,
                                    min_steps=min_steps, subthresh=1.0)


__all__ = ["PreActResNet18", "SHDELife"]
