import torch
import torch.nn as nn
from einops import rearrange
from x_transformers import Encoder, Decoder
import copy

class PatchEmbed(nn.Module):
    """
    Convert images into patches
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        return rearrange(x, 'b e h w -> b (h w) e')

class Predictor(nn.Module):
    """
    Uses VIT to predict target patches from context patches
    """
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.predictor = Decoder(dim=embed_dim, depth=depth, heads=num_heads)

    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim=1)
        return x[:, -target_masks.shape[1]:, :]

class IJEPA_base(nn.Module):
    """
    Main Model for Image to Patch Embedding and Prediction
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, enc_depth, pred_depth, num_heads, post_emb_norm=False, M=4, mode="train", layer_dropout=0.):
        super().__init__()
        self.M = M
        self.mode = mode
        self.layer_dropout = layer_dropout

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_tokens = self.patch_embed.patch_shape[0] * self.patch_embed.patch_shape[1]
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, 0.02)

        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.target_encoder = Encoder(dim=embed_dim, heads=num_heads, depth=enc_depth, layer_dropout=self.layer_dropout)
        self.input_encoder = copy.deepcopy(self.target_encoder).cuda()
        self.predictor = Predictor(embed_dim, num_heads, pred_depth)

    @torch.no_grad() 
    def get_target_block(self, encoder, x, patch_dim, aspect_ratio, scale, M):
        encoder.eval()
        x = self.norm(encoder(x))
        patch_h, patch_w = patch_dim
        num_patches_block = int(patch_h * patch_w * scale)
        block_h = int(torch.sqrt(num_patches_block / aspect_ratio))
        block_w = int(aspect_ratio * block_h)

        target_block = torch.zeros((M, x.shape[0], block_h*block_w, x.shape[2]))
        target_patches, all_patches = [], []

        for z in range(M):
            start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w

            patches = [start_patch + i * patch_w + j for i in range(block_h) for j in range(block_w) if start_patch + i * patch_w + j not in all_patches]
            target_patches.append(patches)
            all_patches.extend(patches)
            target_block[z] = x[:, patches, :]

        return target_block.cuda(), target_patches, all_patches

    def get_context_block(self, x, patch_dim, aspect_ratio, scale, target_patches):
        patch_h, patch_w = patch_dim
        num_patches_block = int(patch_h * patch_w * scale)
        block_h = int(torch.sqrt(num_patches_block / aspect_ratio))
        block_w = int(aspect_ratio * block_h)

        start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w

        patches = [start_patch + i * patch_w + j for i in range(block_h) for j in range(block_w) if start_patch + i * patch_w + j not in target_patches]
        return x[:, patches, :]

    def forward(self, x, target_aspect_ratio=1, target_scale=1, context_aspect_ratio=1, context_scale=1):
        x = self.post_emb_norm(self.patch_embed(x) + self.pos_embedding)
        if self.mode == 'test':
            return self.input_encoder(x)

        target_blocks, target_patches, all_patches = self.get_target_block(self.target_encoder, x, self.patch_embed.patch_shape, target_aspect_ratio, target_scale, self.M)
        context_encoding = self.norm(self.input_encoder(self.get_context_block(x, self.patch_embed.patch_shape, context_aspect_ratio, context_scale, all_patches)))

        m, b, n, e = target_blocks.shape
        prediction_blocks = torch.zeros((m, b, n, e)).cuda()

        for i in range(m):
            target_masks = self.mask_token.repeat(b, n, 1) + self.pos_embedding[:, target_patches[i], :]
            prediction_blocks[i] = self.predictor(context_encoding, target_masks)

        return prediction_blocks, target_blocks