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
        n_ctx = 4
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




def train(model, train_loader, optimizer, scaler, epoch, device, all_train_loss):
    model.train()
    st = time.time()
    op0=[]
    op1=[]
    op2=[]
    tg=[]
    for batch_idx, (data,  target) in enumerate(train_loader):
        data,  target = data.to(device),  target.to(device)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps = torch.randint(20, (3,))
        data = data[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        if rd_ps[1] < 10:
            data = torch.flip(data, dims=[3])

        data = data.float()
        data /= 255
        data[:, 0] -= 0.485
        data[:, 1] -= 0.456
        data[:, 2] -= 0.406
        data[:, 0] /= 0.229
        data[:, 1] /= 0.224
        data[:, 2] /= 0.225
        target-=1
        target /=4

        optimizer.zero_grad()
        with autocast():
            output_0,output_1,output_2 = model(data)
            output2_0=F.softmax(output_0[:,:2])
            output2_1=F.softmax(output_1[:,:2])
            output2_2=F.softmax(output_2[:,:2])

            loss0 = -torch.sum( target[:, 0] * torch.log(output2_0[:, 0]) + (1 - target[:, 0]) * torch.log( output2_0[:, 1])) / output_0.shape[0]
            loss1 = -torch.sum( target[:, 0] * torch.log(output2_1[:, 0]) + (1 - target[:, 0]) * torch.log( output2_1[:, 1])) / output_0.shape[0]
            loss2 = -torch.sum( target[:, 0] * torch.log(output2_2[:, 0]) + (1 - target[:, 0]) * torch.log( output2_2[:, 1])) / output_0.shape[0]
            loss=loss0+loss1+loss2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        all_train_loss.append(loss.item())
        # loss.backward()
        # optimizer.step()
        tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

        op0 = np.concatenate((op0, output2_0[:, 0].detach().cpu().numpy()))
        op1 = np.concatenate((op1, output2_1[:, 0].detach().cpu().numpy()))
        op2 = np.concatenate((op2, output2_2[:, 0].detach().cpu().numpy()))


        # if batch_idx % 100 == 0:
        #     print('Train Epoch:{} [({:.0f}%)]\t Loss: {:.4f}  Loss0: {:.4f} Loss1: {:.4f} Loss2: {:.4f} '.format(
        #         epoch, 100. * batch_idx / len(train_loader), loss.item(), loss0.item(), loss1.item(), loss2.item() ))
    if epoch %100==0:
        print( 'Train ALL Pearson0:', pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="pearson"))
        print( 'Train  ALL Spearman0:', pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="spearman"))
        print( 'Train ALL Pearson1:', pd.Series((op1[::1])).corr((pd.Series(tg[::1])), method="pearson"))
        print( 'Train  ALL Spearman1:', pd.Series((op1[::1])).corr((pd.Series(tg[::1])), method="spearman"))
        print( 'Train ALL Pearson2:', pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="pearson"))
        print( 'Train  ALL Spearman2:', pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="spearman"))

    return all_train_loss


def test(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0

    op0 = []
    op1 = []
    op2 = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data,  target = data.to(device),target.to(device)
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
                output_0, output_1, output_2 = model(data)
                output2_0 = F.softmax(output_0[:, :2])
                output2_1 = F.softmax(output_1[:, :2])
                output2_2 = F.softmax(output_2[:, :2])

                loss0 = -torch.sum( target[:, 0] * torch.log(output2_0[:, 0]) + (1 - target[:, 0]) * torch.log(output2_0[:, 1])) /  output_0.shape[0]
                loss1 = -torch.sum( target[:, 0] * torch.log(output2_1[:, 0]) + (1 - target[:, 0]) * torch.log(output2_1[:, 1])) / output_0.shape[0]
                loss2 = -torch.sum(target[:, 0] * torch.log(output2_2[:, 0]) + (1 - target[:, 0]) * torch.log(output2_2[:, 1])) /  output_0.shape[0]
                loss = loss0 + loss1 + loss2

            all_test_loss.append(loss)
            test_loss += loss
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

            op0 = np.concatenate((op0, output2_0[:, 0].detach().cpu().numpy()))
            op1 = np.concatenate((op1, output2_1[:, 0].detach().cpu().numpy()))
            op2 = np.concatenate((op2, output2_2[:, 0].detach().cpu().numpy()))

            # if batch_idx % 100 == 0:
            #     print('Test Epoch:{} [({:.0f}%)]\t Loss: {:.4f}  Loss0: {:.4f} Loss1: {:.4f} Loss2: {:.4f} '.format(
            #         epoch, 100. * batch_idx / len(test_loader), loss.item(), loss0.item(), loss1.item(), loss2.item()))


    test_loss /= (batch_idx + 1)
    pl0=pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="pearson")
    sr0=pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="spearman")
    pl1=pd.Series((op1[::1])).corr((pd.Series(tg[::1])), method="pearson")
    sr1=pd.Series((op1[::1])).corr((pd.Series(tg[::1])), method="spearman")
    pl2=pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="pearson")
    sr2=pd.Series((op2[::1])).corr((pd.Series(tg[::1])), method="spearman")
    if epoch==-2:
        print('Test ALL Pearson0:',pl0,'Test  ALL Spearman0:', sr0 )
        print('Test ALL Pearson1:', pl1,'Test  ALL Spearman1:', sr1)
        print('Test ALL Pearson2:', pl2,'Test  ALL Spearman2:', sr2)

    return all_test_loss, (pl0+pl1+pl2)/3,(sr0+sr1+sr2)/3


def mysampling (x,nums=3):
    hist,bins=np.histogram(x,bins=nums)
    inds=[]
    for i in range(len(bins)-1):
        inds.append(np.where(((x<bins[i+1])*(x>bins[i]))==1)[0])
    return inds


def main(wt):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    device = torch.device("cuda")

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
    X = np.concatenate((X, Xtest), axis=0)
    Y = np.concatenate((Y, Ytest), axis=0)


    op = sio.loadmat('livew_ds.mat')['distance']
    sel_op = np.zeros_like(op)[:3]

    opt=np.expand_dims(np.mean(op,axis=0),axis=0)
    opt=np.abs(op-np.repeat(opt,5,axis=0))
    indt=np.argsort(opt, axis=0)
    for i in range(50):
        for j in range(op.shape[2]):
            sel_op[:, i, j] =op[ indt[:3,i,j],i,j]
    op=sel_op
    ind = np.argsort(op.sum(0).sum(0))[::-1]
    indt=np.zeros_like(ind)
    indt[ind]=np.arange(len(ind),0,-1)

    all_data = sio.loadmat('prompt3vote_livew.mat')
    op0 = all_data['output1']
    op1 = all_data['output2']
    op2 = all_data['output3']
    op = np.concatenate((np.expand_dims(op0, 2), np.expand_dims(op1, 2), np.expand_dims(op2, 2)), axis=2)
    ind = np.argsort(np.std(op, axis=2).sum(0))[::-1]
    indt2=np.zeros_like(ind)
    indt2[ind] = np.arange(len(ind), 0, -1)
    st = 0.66
    rt = 0.05
    indt2[ind[np.arange(int(len(ind) * st), int(len(ind) * st) + int(len(ind) * rt))]] = np.arange(0,-int(len(ind) * rt),-1)

    indt = wt * indt + indt2
    ind = np.argsort(indt)[::-1]

    # inds_train=np.concatenate((np.arange(0,len(ind)*0.015),np.arange(len(ind)*0.5,len(ind)*0.5+len(ind)*0.025),np.arange(len(ind)*0.9,len(ind)*0.9+len(ind)*0.1)),axis=0).astype(int)
    inds_train = np.arange(len(ind) - len(ind) * rt, len(ind)).astype(int)
    inds_test = np.setdiff1d(ind, inds_train).astype(int)

    Xtest = X[ind[inds_test]]
    Ytest = Y[ind[inds_test]]
    X = X[ind[inds_train]]
    Y = Y[ind[inds_train]]
    best_plccs=[]
    best_srccs=[]
    for i in range(3):
        classnames = [['good','bad' ],['clear','unclear' ],['high quality','low quality']]
        clip_model = load_clip_to_cpu('ViT-B/16').float()
        print("Building custom CLIP")
        model = CustomCLIP( classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in model.named_parameters():

            if name_to_update  in name:
                param.requires_grad_(True)
                # print(name)

            if name_to_update not in name:
                param.requires_grad_(False)

        # # Double check
        # enabled = set()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         enabled.add(name)
        # print(f"Parameters to be updated: {enabled}")


        model = nn.DataParallel(model).to(device)

        # model.load_state_dict(torch.load( 'koniq244_swintiny_adv_5split_'+str(i)+'.pt'))
        ###################################################################


        train_dataset = Mydataset(X, Y)
        test_dataset = Mydataset(Xtest,  Ytest)



        max_plsp=-1
        min_loss = 1e8
        lr = 0.01
        weight_decay = 1e-4
        batch_size = 32*6
        epochs = 2000
        num_workers_train = 0
        num_workers_test = 0
        ct=0


        train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train,pin_memory=True)
        test_loader = DataLoaderX(test_dataset, batch_size=batch_size*18, shuffle=False, num_workers=num_workers_test,pin_memory=True)

        all_train_loss = []
        all_test_loss = []
        all_test_loss,pl,pl2= test(model, test_loader, -1, device, all_test_loss)
        ct = 0
        lr = 0.001
        max_plsp = -2
        scaler =  torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            # print(lr)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

            start = time.time()
            all_train_loss = train(model, train_loader, optimizer, scaler, epoch, device, all_train_loss)
            # print(time.time() - start)
            if epoch%5==4:
                ct += 1
                all_test_loss, pl,pl2 = test(model, test_loader, epoch, device, all_test_loss)
                # print("time:", time.time() - start)
            if epoch == 10:
                lr = 0.003
            if epoch ==20:
                lr = 0.01
            if max_plsp < pl+pl2:
                save_nm = 'livew244_rt25_firststep_'+str(i)+str(wt)+'.pt'
                max_plsp = pl+pl2
                torch.save(model.state_dict(), save_nm)
                ct = 0


            if ct > 5 and epoch > 10:
                model.load_state_dict(torch.load(save_nm))
                lr *= 0.3
                ct = 0
                if lr<5e-5:
                    all_test_loss, pl, pl2= test(model, test_loader, -2, device, all_test_loss)
                    best_plccs.append(pl)
                    best_srccs.append(pl2)

                    print('Weight:', wt,'Split:', i, 'End!','PLCC:', best_plccs, 'SRCC:', best_srccs)
                    break

if __name__ == '__main__':
    main(0.005)
    main(0.01)
    main(0.1)
    main(1)
