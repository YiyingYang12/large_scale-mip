# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import logging
import os
import shutil
from typing import *
import clip

import faulthandler
import torch.nn.functional as F
import gin
import torch
from torch.utils.tensorboard import SummaryWriter   
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from typing import *
from src.data.data_util.nerf_360_v2 import load_nerf_360_v2_data

from src.data.litdata import (
    LitDataBlender,
    LitDataBlenderMultiScale,
    LitDataLF,
    LitDataLLFF,
    LitDataNeRF360V2,
    LitDataRefNeRFReal,
    LitDataShinyBlender,
    LitDataTnT,
)

#from src.model.dvgo.model import LitDVGO
from src.model.mipnerf360.model import LitMipNeRF360
from src.model.mipnerf.model import LitMipNeRF
from src.model.nerf.model import LitNeRF
from src.model.nerfpp.model import LitNeRFPP
from src.model.plenoxel.model import LitPlenoxel
from src.model.refnerf.model import LitRefNeRF

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

#from select_option import select_callback, select_dataset, select_model

def select_callback(model_name):

    callbacks = []

    if model_name == "plenoxel":
        import src.model.plenoxel.model as model

        callbacks += [model.ResampleCallBack()]

    if model_name == "dvgo":
        import src.model.dvgo.model as model

        callbacks += [
            model.Coarse2Fine(),
            model.ProgressiveScaling(),
            model.UpdateOccupancyMask(),
        ]

    return callbacks
def select_dataset(
    dataset_name: str,
    datadir: str,
    scene_name: str,
    ):
    if dataset_name == "blender":
        data_fun = LitDataBlender
    elif dataset_name == "blender_multiscale":
        data_fun = LitDataBlenderMultiScale
    elif dataset_name == "llff":
        data_fun = LitDataLLFF
    elif dataset_name == "tanks_and_temples":
        data_fun = LitDataTnT
    elif dataset_name == "lf":
        data_fun = LitDataLF
    elif dataset_name == "nerf_360_v2":
        data_fun = LitDataNeRF360V2
    elif dataset_name == "shiny_blender":
        data_fun = LitDataShinyBlender
    elif dataset_name == "refnerf_real":
        data_fun = LitDataRefNeRFReal

    return data_fun(
        datadir=datadir,
        scene_name=scene_name,
    )
def select_model(
    model_name: str,
    ):

    if model_name == "nerf":
        return LitNeRF()
    elif model_name == "mipnerf":
        return LitMipNeRF()
    elif model_name == "plenoxel":
        return LitPlenoxel()
    elif model_name == "nerfpp":
        return LitNeRFPP()
    #elif model_name == "dvgo":
    #    return LitDVGO()
    elif model_name == "refnerf":
        return LitRefNeRF()
    elif model_name == "mipnerf360":
        return LitMipNeRF360()

    else:
        raise f"Unknown model named {model_name}"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")
'''
def get_embed_fn(model_type, num_layers=-1, spatial=False, checkpoint=False, clip_cache_root=None):
    if model_type.startswith('clip_'):
        if model_type == 'clip_rn50':
            assert clip_cache_root
            clip_utils.load_rn(jit=False, root=clip_cache_root)
            if spatial:
                _clip_dtype = clip_utils.clip_model_rn.clip_model.dtype
                assert num_layers == -1
                def embed(ims):
                    ims = clip_utils.CLIP_NORMALIZE(ims).type(_clip_dtype)
                    return clip_utils.clip_model_rn.clip_model.visual.featurize(ims)  # [N,C,56,56]
            else:
                embed = lambda ims: clip_utils.clip_model_rn(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers).unsqueeze(1)
            assert not clip_utils.clip_model_rn.training
        elif model_type.startswith('clip_vit'):
            assert clip_cache_root
            if model_type == 'clip_vit':
                clip_utils.load_vit(root=clip_cache_root)
            elif model_type == 'clip_vit_b16':
                clip_utils.load_vit('ViT-B/16', root=clip_cache_root)
            if spatial:
                def embed(ims):
                    emb = clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers)  # [N,L=50,D]
                    return emb[:, 1:].view(emb.shape[0], 7, 7, emb.shape[2]).permute(0, 3, 1, 2)  # [N,D,7,7]
            else:
                embed = lambda ims: clip_utils.clip_model_vit(images_or_text=clip_utils.CLIP_NORMALIZE(ims), num_layers=num_layers)  # [N,L=50,D]
            assert not clip_utils.clip_model_vit.training
        elif model_type == 'clip_rn50x4':
            assert not spatial
            clip_utils.load_rn(name='RN50x4', jit=False)
            assert not clip_utils.clip_model_rn.training
            embed = lambda ims: clip_utils.clip_model_rn(images_or_text=clip_utils.CLIP_NORMALIZE(ims), featurize=False)
    elif model_type.startswith('timm_'):
        assert num_layers == -1
        assert not spatial

        model_type = model_type[len('timm_'):]
        encoder = timm.create_model(model_type, pretrained=True, num_classes=0)
        encoder.eval()
        normalize = torchvision.transforms.Normalize(
            encoder.default_cfg['mean'], encoder.default_cfg['std'])  # normalize an image that is already scaled to [0, 1]
        encoder = nn.DataParallel(encoder).to(device)
        embed = lambda ims: encoder(normalize(ims)).unsqueeze(1)
    elif model_type.startswith('torch_'):
        assert num_layers == -1
        assert not spatial

        model_type = model_type[len('torch_'):]
        encoder = torch.hub.load('pytorch/vision:v0.6.0', model_type, pretrained=True)
        encoder.eval()
        encoder = nn.DataParallel(encoder).to(device)
        normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize an image that is already scaled to [0, 1]
        embed = lambda ims: encoder(normalize(ims)).unsqueeze(1)
    else:
        raise ValueError

    if checkpoint:
        return lambda x: run_checkpoint(embed, x)

    return embed
'''


@gin.configurable()

def run(
    ginc: str,
    ginb: str,
    resume_training: bool,
    ckpt_path: Optional[str],
    scene_name: Optional[str],
    datadir: Optional[str] = None,
    logbase: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    postfix: Optional[str] = None,
    entity: Optional[str] = None,
    # Optimization
    max_steps: int = -1,
    max_epochs: int = -1,
    precision: int = 32,
    # Logging
    log_every_n_steps: int = 1000,
    progressbar_refresh_rate: int = 5,
    # Run Mode
    run_train: bool = True,
    run_eval: bool = True,
    run_render: bool = True,
    num_devices: Optional[int] = None,
    num_sanity_val_steps: int = 0,
    seed: int = 777,
    debug: bool = False,
    save_last: bool = True,
    grad_max_norm=0.0,
    grad_clip_algorithm="norm",
):

   # load codebook
    
    '''
    vq_path = "/root/NeRF-Factory-main/NeRF-Factory-main/checkpoints/code.ckpt"
    code_data = torch.load(vq_path, map_location="cpu")
    state_dict = code_data.get("state_dict", {})
    codebook = state_dict["quantize.embedding.weight"]
    '''
    logging.getLogger("lightning").setLevel(logging.ERROR)
    
    datadir = datadir.rstrip("/")

    exp_name = (
        model_name + "_" + dataset_name + "_" + scene_name + "_" + str(seed).zfill(3)
    )
    if postfix is not None:
        exp_name += "_" + postfix
    if debug:
        exp_name += "_debug"

    if num_devices is None:
        num_devices = torch.cuda.device_count()

    if model_name in ["plenoxel"]:
        num_devices = 1

    if logbase is None:
        logbase = "logs"

    os.makedirs(logbase, exist_ok=True)
    logdir = os.path.join(logbase, exp_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, exp_name), exist_ok=True)

    logger = pl_loggers.TensorBoardLogger(
        save_dir=logdir,
        name=exp_name,
    )
    faulthandler.enable()
    # Logging all parameters
    if run_train:
        txt_path = os.path.join(logdir, "config.gin")
        with open(txt_path, "w") as fp_txt:
            for config_path in ginc:
                fp_txt.write(f"Config from {config_path}\n\n")
                with open(config_path, "r") as fp_config:
                    readlines = fp_config.readlines()
                for line in readlines:
                    fp_txt.write(line)
                fp_txt.write("\n")

            fp_txt.write("\n### Binded options\n")
            for line in ginb:
                fp_txt.write(line + "\n")

    seed_everything(seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="val/psnr",
        dirpath=logdir,
        filename="best",
        save_top_k=1,
        mode="max",
        save_last=save_last,
    )
    tqdm_progrss = TQDMProgressBar(refresh_rate=progressbar_refresh_rate)

    callbacks = []
    if not model_name in ["plenoxel"]:
        callbacks.append(lr_monitor)
    callbacks += [model_checkpoint, tqdm_progrss]
    callbacks += select_callback(model_name)

    ddp_plugin = DDPStrategy(find_unused_parameters=True) if num_devices > 1 else None

    trainer = Trainer(
        logger=logger if run_train else None,
        log_every_n_steps=log_every_n_steps,
        devices=num_devices,
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="gpu",
        replace_sampler_ddp=False,
        strategy=ddp_plugin,
        check_val_every_n_epoch=1,
        precision=precision,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=callbacks,
        gradient_clip_algorithm=grad_clip_algorithm,
        gradient_clip_val=grad_max_norm,
    )

    if resume_training:
        if ckpt_path is None:
            ckpt_path = f"{logdir}/last.ckpt"

    data_module = select_dataset(
        dataset_name=dataset_name,
        scene_name=scene_name,
        datadir=datadir,
    )
    #xuanq
    model = select_model(model_name=model_name)


    
    model.logdir = logdir
    if run_train:
        best_ckpt = os.path.join(logdir, "best.ckpt")
        if os.path.exists(best_ckpt):
            os.remove(best_ckpt)
        version0 = os.path.join(logdir, exp_name, "version_0")
        if os.path.exists(version0):
            shutil.rmtree(version0, True)

        trainer.fit(model, data_module, ckpt_path=ckpt_path)

    if run_eval:
        ckpt_path = (
            f"{logdir}/best.ckpt"
            if model_name != "mipnerf360"
            else f"{logdir}/last.ckpt"
        )
        trainer.test(model, data_module, ckpt_path=ckpt_path)

    if run_render:
        ckpt_path = (
            f"{logdir}/best.ckpt"
            if model_name != "mipnerf360"
            else f"{logdir}/last.ckpt"
        )
        trainer.predict(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    parser.add_argument(
        "--resume_training",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="gin bindings",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="path to checkpoints"
    )
    parser.add_argument(
        "--scene_name", type=str, default=None, help="scene name to render"
    )
    parser.add_argument("--seed", type=int, default=220901, help="seed to use")
    '''
     # Consistency model arguments
    parser.add_argument("--consistency_model_type", type=str, default='clip_vit') # choices=['clip_vit', 'clip_vit_b16', 'clip_rn50']
    parser.add_argument("--consistency_model_num_layers", type=int, default=-1)
    parser.add_argument("--clip_cache_root", type=str, default=os.path.expanduser("~/.cache/clip"))
    parser.add_argument("--consistency_size", type=int, default=224)
    parser.add_argument("--checkpoint_embedding", action='store_true')
    parser.add_argument("--pixel_interp_mode", type=str, default='bicubic')
    '''
    #parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    #torch.cuda.set_device(args.local_rank)
    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)

    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {ginbs}")

    gin.parse_config_files_and_bindings(args.ginc, ginbs)
    run(
        ginc=args.ginc,
        ginb=ginbs,
        scene_name=args.scene_name,
        resume_training=args.resume_training,
        ckpt_path=args.ckpt_path,
        seed=args.seed,
    )
