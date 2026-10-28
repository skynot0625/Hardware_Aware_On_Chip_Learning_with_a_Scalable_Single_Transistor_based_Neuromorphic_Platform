"""Select original research classes without wrapping or rewriting their forward methods."""

from torch import nn

from . import model, model_current, model_current_eLIFE, model_eLIFE, model_shd


def PreActResNet18(scale: int = 64, num_steps: int = 5, elife: bool = False,
                 surrogate: str = "csg") -> nn.Module:
    classes = {
        ("sg", False): model.SOLO_PreactResNet18,
        ("csg", False): model_current.SOLO_PreactResNet18,
        ("sg", True): model_eLIFE.SOLO_PreactResNet18,
        ("csg", True): model_current_eLIFE.SOLO_PreactResNet18,
    }
    if (surrogate, elife) not in classes or scale < 1 or num_steps < 1:
        raise ValueError("require sg/csg, positive scale and positive num_steps")
    return classes[surrogate, elife](scale=scale, num_steps=num_steps,
                                    init_LIF_tau=1.0, init_LI_tau=3.0, subthresh=1.0)


def SHDELife(scale: int = 1024, num_steps: int = 20, min_steps: int = 4,
             surrogate: str = "csg", elife: bool = True) -> nn.Module:
    if scale != 1024 or num_steps != 20 or not 2 <= min_steps <= 20:
        raise ValueError("original SHD classes require scale=1024, num_steps=20 and 2<=min_steps<=20")
    classes = {
        ("sg", False): model_shd.mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3_pure,
        ("csg", False): model_shd.mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3_pure,
        ("sg", True): model_shd.mlp4_SOLOplus_smnist_RES_RSNN_ht_res_fb3,
        ("csg", True): model_shd.mlp4_SOLOplus_current_smnist_RES_RSNN_ht_res_fb3,
    }
    if (surrogate, elife) not in classes:
        raise ValueError("surrogate must be sg or csg")
    return classes[surrogate, elife](scale=scale, num_steps=num_steps,
                                    min_steps=min_steps, subthresh=1.0)


__all__ = ["PreActResNet18", "SHDELife"]
