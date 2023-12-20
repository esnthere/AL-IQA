import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import math
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
from imgaug import augmenters as iaa
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import time
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from skimage import io
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from functools import partial

import matplotlib.pyplot as plt
import lmdb
from prefetch_generator import BackgroundGenerator

from torch.cuda.amp import autocast as autocast


_tokenizer = _Tokenizer()


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, class_token_position):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 8
        ctx_init = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        CSC = False
        self.class_token_position = class_token_position
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner0 = PromptLearner(classnames[0], clip_model, 'end')
        self.tokenized_prompts0 = self.prompt_learner0.tokenized_prompts

        self.prompt_learner1 = PromptLearner(classnames[1], clip_model, 'end')
        self.tokenized_prompts1 = self.prompt_learner1.tokenized_prompts

        self.prompt_learner2 = PromptLearner(classnames[2], clip_model, 'end')
        self.tokenized_prompts2 = self.prompt_learner2.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        text_features0 = self.text_encoder(self.prompt_learner0(), self.tokenized_prompts0)
        text_features0 = text_features0 / text_features0.norm(dim=-1, keepdim=True)
        logits0 = logit_scale * image_features @ text_features0.t()

        text_features1 = self.text_encoder(self.prompt_learner1(), self.tokenized_prompts1)
        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
        logits1 = logit_scale * image_features @ text_features1.t()

        text_features2 = self.text_encoder(self.prompt_learner2(), self.tokenized_prompts2)
        text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
        logits2 = logit_scale * image_features @ text_features2.t()

        return logits0, logits1, logits2


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)
    def __getitem__(self, index):
        return torch.from_numpy(self.imgs[index]),self.labels[index]
    def __len__(self):
        return (self.imgs).shape[0]






def test(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    op2 = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data,  target = data.to(device),target.to(device)
            if data.shape[2]==244:
                data = data[:, :,10:10 + 224, 10:10 + 224]

            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            target -= 1
            target /= 4
            with autocast():
                _, _, output_2 = model(data)
                output2_2 = F.softmax(output_2[:, :2])

            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))
            op2 = np.concatenate((op2, output2_2[:, 0].detach().cpu().numpy()))

    pl2=pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="pearson")
    sr2=pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="spearman")
    print('Test ALL Pearson:', pl2, 'Test  ALL Spearman:', sr2)
    return all_test_loss, pl2,sr2


def mymodel(device):

    classnames = [['good','bad' ],['clear','unclear' ],['high quality','low quality']]
    clip_model = load_clip_to_cpu('ViT-B/16').float()
    print("Building custom CLIP")
    model = CustomCLIP( classnames, clip_model)
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load( 'koniq_rt80.pt'))


    return  model

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
    device = torch.device("cuda")

    ################
    print("KonIQ Test Results:")

    all_data = sio.loadmat('E:\Database\IQA Database\## KonIQ-10k Image Database\Koniq_224.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)

    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    del all_data
    X = np.concatenate((X, Xtest), axis=0)
    Y = np.concatenate((Y, Ytest), axis=0)

    all_data = sio.loadmat('koniq_rt80_index.mat')
    ind_test = all_data['ind_test'][0]

    Xtest = X[ind_test]
    Ytest = Y[ind_test]
    ##################
    model = mymodel(device)
    test_dataset = Mydataset(Xtest, Ytest)
    test_loader = DataLoaderX(test_dataset, batch_size=32 * 8, shuffle=False, num_workers=0, pin_memory=True)
    all_test_loss, pl, pl2 = test(model, test_loader, -1, device, [])

    ###################################################################
    print("SPAQ Test Results:")

    all_data = sio.loadmat('E:\Database\SPAQ\spaq_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y.reshape(Y.shape[1], 1)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest.reshape(Ytest.shape[1], 1)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoaderX(test_dataset, batch_size=32*18, shuffle=False, num_workers=0,pin_memory=True)
    all_test_loss,pl,pl2= test(model, test_loader, -1, device, [])
    ################################################################
    print("LIVEW Test Results:")

    all_data = sio.loadmat('E:\Database\LIVEW\livew_244.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y.reshape(Y.shape[0], 1)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoaderX(test_dataset, batch_size=32 * 18, shuffle=False, num_workers=0, pin_memory=True)
    all_test_loss, pl, pl2 = test(model, test_loader, -1, device, [])
    ################################################################
    print("LIVEW Test Results:")

    all_data = sio.loadmat('E:\Database\LIVEW\livew_224.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y.reshape(Y.shape[0], 1)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoaderX(test_dataset, batch_size=32 * 18, shuffle=False, num_workers=0, pin_memory=True)
    all_test_loss, pl, pl2 = test(model, test_loader, -1, device, [])
 ###################################################################
    print("RBID Test Results:")

    all_data = sio.loadmat('E:\Database\RBID\\rbid_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y.reshape(Y.shape[0], 1)
    Y = Y * 0.8 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    Ytest = Ytest * 0.8 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoaderX(test_dataset, batch_size=32 * 18, shuffle=False, num_workers=0, pin_memory=True)
    all_test_loss, pl, pl2 = test(model, test_loader, -1, device, [])

################################################################
    print("CID Test Results:")

    all_data = sio.loadmat('E:\Database\CID2013\\cid_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = (Y + 0) / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = (Ytest + 0) / 25 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoaderX(test_dataset, batch_size=32 * 18, shuffle=False, num_workers=0, pin_memory=True)
    all_test_loss, pl, pl2 = test(model, test_loader, -1, device, [])

if __name__ == '__main__':
    main()
