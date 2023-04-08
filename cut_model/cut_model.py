import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from cut_networks import *

class CUT_model():
    def __init__(self, temperature=0.07, lambda_x=1.0, lambda_y=1.0, v_k_dim=256, v_layer_dims=[3,128,256,256,256,256], n_patches=256, device="cuda"):
        self.g_encoder = CUT_G_encoder().to(device=device)
        self.g_decoder = CUT_G_decoder().to(device=device)
        self.h_model = CUT_H(v_k_dim, input_channels=v_layer_dims).to(device=device)
        self.d_model = CUT_D().to(device=device)

        self.gan_criterion = nn.MSELoss()
        self.patchNCE_criterion = []

        for i in v_layer_dims:
            self.patchNCE_criterion.append(PatchNCELoss(n_patches, temperature, device))

        self.g_optimizer = torch.optim.AdamW(nn.ModuleList([self.g_encoder, self.g_decoder]).parameters(), lr=0.0002)
        self.d_optimizer = torch.optim.AdamW(self.d_model.parameters(), lr=0.0002)
        self.h_optimizer = torch.optim.AdamW(self.h_model.parameters(), lr=0.005)

        self.g_scheduler = torch.optim.lr_scheduler.LinearLR(self.g_optimizer, start_factor=1.0, end_factor=0.01)
        self.d_scheduler = torch.optim.lr_scheduler.LinearLR(self.d_optimizer, start_factor=1.0, end_factor=0.01)
        self.h_scheduler = torch.optim.lr_scheduler.LinearLR(self.h_optimizer, start_factor=1.0, end_factor=0.01)

        self.lambda_x = lambda_x
        self.lambda_y = lambda_y
        self.n_patches = n_patches
        self.v_layer_dims = v_layer_dims

        self.device = device

        #TODO: derive size from d_model output
        self.tgt_ones = torch.ones((1,1,30,30), device=self.device)
        self.tgt_zeros = torch.zeros((1,1,30,30), device=self.device)
    
    #TODO: modify for FASTCUT
    def forward(self, x, y, is_train=True):
        if is_train:
            self.g_encoder.train()
            self.g_decoder.train()
            self.h_model.train()
            self.d_model.train()
        else:
            self.g_encoder.eval()
            self.g_decoder.eval()
            self.h_model.eval()
            self.d_model.eval()

        #create fake imgs from X domain with sample x
        #and Y domain with sample y
        z_x_feats, z_x = self.g_encoder(x)
        y_hat = self.g_decoder(z_x)

        #return only imgs (TODO: current returns identity too, maybe not?)
        if not is_train:
            return y_hat
        
        z_y_feats, z_y = self.g_encoder(y)
        y_idt = self.g_decoder(z_y)

        #get feats from intermediate layers for fake images too
        z_y_hat_feats, _ = self.g_encoder(y_hat)
        z_y_idt_feats, _ = self.g_encoder(y_idt)

        #create random permutation for patch_ids for this forward pass
        rand_perms = []
        for i in range(len(self.v_layer_dims)):
            n = z_x_feats[i].shape[2] * z_x_feats[i].shape[3]
            rand_perms.append(torch.randperm(n, device=self.device)[:self.n_patches])
            
        #encode to latent space
        h_x = self.h_model(z_x_feats, rand_perms)
        h_y = self.h_model(z_y_feats, rand_perms)
        h_y_hat = self.h_model(z_y_hat_feats, rand_perms)
        h_y_idt = self.h_model(z_y_idt_feats, rand_perms)

        return y_hat, y_idt, h_x, h_y, h_y_hat, h_y_idt

    def optimize(self, y, y_hat, h_x, h_y, h_y_hat, h_y_idt):

        #optimize discriminator
        self.d_optimizer.zero_grad()
        for p in self.d_model.parameters():
            p.requires_grad = True

        y_d = self.d_model(y)
        y_hat_d = self.d_model(y_hat.detach())

        gan_loss_y = self.gan_criterion(y_d, torch.ones_like(y_d, device=self.device)).mean()
        gan_loss_y_hat = self.gan_criterion(y_hat_d, torch.zeros_like(y_hat_d, device=self.device)).mean()

        d_gan_loss = (gan_loss_y + gan_loss_y_hat) * 0.5
        
        d_gan_loss.backward()
        detached_d_loss = d_gan_loss.item()
        self.d_optimizer.step()
        self.d_scheduler.step()

        #optimize generator (including patchNCE)
        self.g_optimizer.zero_grad()
        self.h_optimizer.zero_grad()

        for p in self.d_model.parameters():
            p.requires_grad = False

        y_hat_d = self.d_model(y_hat)
        gan_loss_y_hat = self.gan_criterion(y_hat_d, torch.ones_like(y_hat_d, device=self.device)).mean()

        nce_loss_x = 0.0
        nce_loss_idt = 0.0

        for i, crit in enumerate(self.patchNCE_criterion):
            nce_loss_x += crit(h_y_hat[i], h_x[i]).mean()
            nce_loss_idt += crit(h_y_idt[i], h_y[i]).mean()

        nce_loss_x /= len(self.patchNCE_criterion)
        nce_loss_idt /= len(self.patchNCE_criterion)

        h_patch_loss = (((self.lambda_x * nce_loss_x) + (self.lambda_y * nce_loss_idt)) * 0.5)
        g_gan_loss = gan_loss_y_hat 

        total_gan_loss = g_gan_loss + h_patch_loss
        
        total_gan_loss.backward() #this will propagate through h_model too
        self.g_optimizer.step()
        self.h_optimizer.step()

        self.g_scheduler.step()
        self.h_scheduler.step()

        return detached_d_loss, total_gan_loss.item()


    def save(self, output_dir, suffix="0"):
        torch.save(self.g_encoder.state_dict(), os.path.join(output_dir, f"g_enc_{suffix}.pt"))
        torch.save(self.g_decoder.state_dict(), os.path.join(output_dir, f"g_dec_{suffix}.pt"))
        torch.save(self.d_model.state_dict(), os.path.join(output_dir, f"d_model_{suffix}.pt"))
        torch.save(self.h_model.state_dict(), os.path.join(output_dir, f"h_model_{suffix}.pt"))

    def load(self, model_dir, suffix=None):

        if suffix is None:
            suffix = ""

        self.g_encoder.load_state_dict(torch.load(os.path.join(model_dir,f"g_enc_{suffix}.pt")))
        self.g_decoder.load_state_dict(torch.load(os.path.join(model_dir,f"g_dec_{suffix}.pt")))
        self.d_model.load_state_dict(torch.load(os.path.join(model_dir,f"d_model_{suffix}.pt")))
        self.h_model.load_state_dict(torch.load(os.path.join(model_dir,f"h_model_{suffix}.pt")))



