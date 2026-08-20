import sys
from types import MethodType

import torch
import coremltools as ct
from transformers import AutoModelForDepthEstimation

W = int(sys.argv[1]) if len(sys.argv) > 1 else 392
H = int(sys.argv[2]) if len(sys.argv) > 2 else 294
assert W % 14 == 0 and H % 14 == 0

model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
).eval()

# Precompute the (bicubic) interpolated position encoding for the fixed input
# size, so the trace contains it as a constant. coremltools has no bicubic op.
emb = model.backbone.embeddings
orig_interp = type(emb).interpolate_pos_encoding
with torch.no_grad():
    num_patches = (H // 14) * (W // 14)
    dim = emb.position_embeddings.shape[-1]
    fake = torch.zeros(1, num_patches + 1, dim)
    PE = orig_interp(emb, fake, H, W)


def patched(self, embeddings, height, width):
    return PE


emb.interpolate_pos_encoding = MethodType(patched, emb)

MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor([x for x in MEAN]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([x for x in STD]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, image):
        # image arrives as 0..255 RGB (CoreML ImageType default)
        x = (image - self.mean) / self.std
        depth = self.m(x, return_dict=False)[0]  # (1, H, W)
        depth = depth / depth.max()  # normalize to [0, 1] like Apple's models
        return depth.unsqueeze(0)  # (1, 1, H, W) grayscale image


wrapped = Wrapper(model).eval()
example = torch.rand(1, 3, H, W) * 255.0
with torch.no_grad():
    traced = torch.jit.trace(wrapped, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, H, W), color_layout=ct.colorlayout.RGB)],
    outputs=[ct.ImageType(name="depth", color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,
    convert_to="mlprogram",
)
out = f"DepthAnythingV2Small_{W}x{H}_F16.mlpackage"
mlmodel.save(out)
print("saved", out)
