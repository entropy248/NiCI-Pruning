from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DModel
import torch_pruning as tp
import torch
from torch_pruning.pruner import function
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import os
from glob import glob
from PIL import Image
import accelerate
import utils
from taylor_step_pruner import TaylorStepPruner

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,  default="celeba", help="path to an image folder")
parser.add_argument("--model_path", type=str, default="pretrained/ddpm_ema_cifar10/ddpm_ema_cifar10")
parser.add_argument("--iter_num", type=int, default=1)
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--save_path", type=str, default="run/pruned/ddpm_cifar10_pruned")
parser.add_argument("--pruning_ratio", type=float, default=0.3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--device", type=str, default='cuda:4')
parser.add_argument("--pruner", type=str, default='c2c', choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning', 'c2c'])
parser.add_argument("--tau", type=float, default=1.0, help="kd temperature")
parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

args = parser.parse_args()

batch_size = args.batch_size
dataset = args.dataset


class TaylorStepImportance(tp.importance.MagnitudeImportance):
    def __init__(self,
                 group_reduction: str = "max",
                 normalizer: str = 'max',
                 multivariable: bool = False,
                 bias=False,
                 target_types: list = [nn.modules.conv._ConvNd, nn.Linear, nn.modules.batchnorm._BatchNorm]):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.multivariable = multivariable
        self.target_types = target_types
        self.bias = bias

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, step=False):
        group_imp = []
        group_idxs = []
        for i, (dep, idxs) in enumerate(group):
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler
            root_idxs = group[i].root_idxs

            if not isinstance(layer, tuple(self.target_types)):
                continue

            if prune_fn in [
                function.prune_conv_out_channels,
                function.prune_linear_out_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = layer.weight.data.transpose(1, 0)[idxs].flatten(1)
                    dw = layer.weight.grad.data.transpose(1, 0)[
                        idxs].flatten(1)
                else:
                    w = layer.weight.data[idxs].flatten(1)
                    dw = layer.weight.grad.data[idxs].flatten(1)
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

                if self.bias and layer.bias is not None:
                    b = layer.bias.data[idxs]
                    db = layer.bias.grad.data[idxs]
                    local_imp = (b * db).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)


            # Conv in_channels
            elif prune_fn in [
                function.prune_conv_in_channels,
                function.prune_linear_in_channels,
            ]:
                if hasattr(layer, "transposed") and layer.transposed:
                    w = (layer.weight).flatten(1)[idxs]
                    dw = (layer.weight.grad).flatten(1)[idxs]
                else:
                    w = (layer.weight).transpose(0, 1).flatten(1)[idxs]
                    dw = (layer.weight.grad).transpose(0, 1).flatten(1)[idxs]
                if self.multivariable:
                    local_imp = (w * dw).sum(1).abs()
                else:
                    local_imp = (w * dw).abs().sum(1)
                group_imp.append(local_imp)
                group_idxs.append(root_idxs)

            # BN
            elif prune_fn == function.prune_groupnorm_out_channels:
                # regularize BN
                if layer.affine:
                    w = layer.weight.data[idxs]
                    dw = layer.weight.grad.data[idxs]
                    local_imp = (w * dw).abs()
                    group_imp.append(local_imp)
                    group_idxs.append(root_idxs)

                    if self.bias and layer.bias is not None:
                        b = layer.bias.data[idxs]
                        db = layer.bias.grad.data[idxs]
                        local_imp = (b * db).abs()
                        group_imp.append(local_imp)
                        group_idxs.append(root_idxs)

        group_imp = self._reduce(group_imp, group_idxs)
        group_imp = self._normalize(group_imp, self.normalizer)

        return group_imp


if __name__=='__main__':
    # loading images for gradient-based pruning
    if args.pruner in ['taylor', 'diff-pruning', 'c2c']:
        # if args.dataset == 'celeba':
        #     dataset = load_dataset("huggan/CelebA-HQ", split='train')
        dataset = utils.get_dataset(args.dataset)
        print(f"Dataset size: {len(dataset)}")
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
        )

    # Loading pretrained model
    print("Loading pretrained model from {}".format(args.model_path))
    pipeline = DDPMPipeline.from_pretrained(args.model_path).to(args.device)
    scheduler = pipeline.scheduler

    model = pipeline.unet.eval()

    if 'cifar' in args.model_path:
        example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}
    else:
        example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(args.device), 'timestep': torch.ones((1,)).long().to(args.device)}

    if args.pruning_ratio>0:
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance(multivariable=True) # standard first-order taylor expansion
        elif args.pruner == 'random' or args.pruner=='reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(multivariable=False) # a modified version, estimating the accumulated error of weight removal
        elif args.pruner == 'c2c':
            # imp = TaylorRImportance(multivariable=False)
            imp = tp.importance.TaylorImportance(multivariable=False)
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]
        channel_groups = {}

        # pruner = tp.pruner.MagnitudePruner(
        #     model,
        #     example_inputs,
        #     importance=imp,
        #     iterative_steps=1,
        #     # global_pruning=True,
        #     channel_groups=channel_groups,
        #     ch_sparsity=args.pruning_ratio,
        #     ignored_layers=ignored_layers,
        # )

        pruner = TaylorStepPruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            # global_pruning=True,
            channel_groups=channel_groups,
            ch_sparsity=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )
        pruner.set_storage()

        # for name, node in pruner.DG.module2node.items():
        #     print(name, node)

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()
        import random

        if args.pruner in ['taylor', 'diff-pruning', 'c2c']:
            print("Accumulating gradients for pruning...")

            flag = True
            for step, batch in enumerate(train_dataloader):

                model.zero_grad()

                if isinstance(batch, (list, tuple)):
                    clean_images = batch[0]
                else:
                    clean_images = batch
                clean_images = clean_images.to(args.device)

                noise = torch.randn(clean_images.shape).to(clean_images.device)

                w_t = 0

                if args.pruner == 'c2c':
                    loss_max = 0
                    step_k = 0
                    while step_k < 1000:
                        timesteps = (step_k * torch.ones((args.batch_size,), device=clean_images.device)).long()
                        alpha_t = scheduler.alphas_cumprod[step_k]

                        model_output = model(clean_images, timesteps).sample
                        loss = torch.nn.functional.mse_loss(model_output, clean_images)
                        print("iter:{} timestep:{} loss:{}".format(step, step_k, loss), end="\r")
                        loss.backward()

                        if step_k < 250:
                            step_k += 2
                        elif 250 <= step_k < 500:
                            step_k += 2 ** 2
                        elif 500 <= step_k < 750:
                            step_k += 2 ** 3
                        elif 750 <= step_k < 1000:
                            step_k += 2 ** 4

                    pruner.store_importance(is_first=flag)
                    flag = False

                else:
                    for step_k in range(1000):
                        timesteps = (step_k * torch.ones((args.batch_size,), device=clean_images.device)).long()
                        noisy_images = scheduler.add_noise(clean_images, noise, timesteps)
                        model_output = model(noisy_images, timesteps).sample
                        loss = torch.nn.functional.mse_loss(model_output, noise)

                        print("iter:{} timestep:{} loss:{} w_t:{}".format(step, step_k, loss, w_t), end="\r")

                        loss.backward()

                        # break
                        if args.pruner in ['diff-pruning']:
                            if loss > loss_max: loss_max = loss
                            if loss < loss_max * args.thr: break # taylor expansion over pruned timesteps ( L_t / L_max > thr )
                    pruner.store_importance(is_first=flag)
                flag = False
                # assert 0

                if args.iter_num != 0 and step + 1 >= args.iter_num:
                    break

        for g in pruner.step(interactive=True):
            g.prune()

        # Update static attributes
        from diffusers.models.resnet import Upsample2D, Downsample2D
        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels == m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
        model.zero_grad()
        del pruner

        if args.pruner=='reinit':
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            reset_parameters(model)

    pipeline.save_pretrained(args.save_path)
    if args.pruning_ratio>0:
        os.makedirs(os.path.join(args.save_path, "pruned"), exist_ok=True)
        torch.save(model, os.path.join(args.save_path, "pruned", "unet_pruned.pth"))

    # Sampling images from the pruned model
    pipeline = DDIMPipeline(
        unet = model,
        scheduler = DDIMScheduler.from_pretrained(args.save_path, subfolder="scheduler")
    )
    with torch.no_grad():
        generator = torch.Generator(device=pipeline.device).manual_seed(0)
        pipeline.to("cuda")
        images = pipeline(num_inference_steps=100, batch_size=args.batch_size, generator=generator, output_type="numpy").images
        os.makedirs(os.path.join(args.save_path, 'vis'), exist_ok=True)
        torchvision.utils.save_image(torch.from_numpy(images).permute([0, 3, 1, 2]), "{}/vis/after_pruning.png".format(args.save_path))

