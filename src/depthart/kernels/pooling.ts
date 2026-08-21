// @ts-nocheck - vendored from software-mansion/TypeGPU (upstream-typechecked)
import { d, std, tgpu } from 'typegpu';
import { blockedElement, hwc4Index, maskPaddedChannels } from './helpers.ts';
import { poolLayout } from './layouts.ts';
import { DEPTH_KERNEL_WORKGROUP_SIZE } from './types.ts';

/** Fixed-window average pool. DepthART uses non-padded 8/4/2 windows and equal strides */
export const averagePoolKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [DEPTH_KERNEL_WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const index = gid.x;
  const params = poolLayout.$.params;
  if (index >= params.elementCount) {
    return;
  }

  const output = blockedElement(index, params.outputWidth, params.channelBlocks);
  const inputOrigin = d.vec2u(output.x * params.strideX, output.y * params.strideY);
  let sum = d.vec4f(0);

  for (let ky = d.u32(0); ky < params.windowHeight; ky += 1) {
    for (let kx = d.u32(0); kx < params.windowWidth; kx += 1) {
      const inputX = inputOrigin.x + kx;
      const inputY = inputOrigin.y + ky;
      if (inputX < params.inputWidth && inputY < params.inputHeight) {
        sum +=
          poolLayout.$.src[
            hwc4Index(inputY, inputX, output.z, params.inputWidth, params.channelBlocks)
          ];
      }
    }
  }

  const divisor = d.f32(params.windowWidth * params.windowHeight);
  poolLayout.$.dst[index] = maskPaddedChannels(sum / divisor, output.z, params.logicalChannels);
});

// LIGHTDEMO PATCH: fixed-window max pool - replaces the converter's one-hot
// shift + running-max decomposition (~6 dispatches) with one dispatch. Out-
// of-bounds taps are simply skipped (VALID pooling); padded channel lanes
// are zeroed like the average pool's.
export const maxPoolKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [DEPTH_KERNEL_WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const index = gid.x;
  const params = poolLayout.$.params;
  if (index >= params.elementCount) {
    return;
  }

  const output = blockedElement(index, params.outputWidth, params.channelBlocks);
  const inputOrigin = d.vec2u(output.x * params.strideX, output.y * params.strideY);
  let best = d.vec4f(-3.0e38);

  for (let ky = d.u32(0); ky < params.windowHeight; ky += 1) {
    for (let kx = d.u32(0); kx < params.windowWidth; kx += 1) {
      const inputX = inputOrigin.x + kx;
      const inputY = inputOrigin.y + ky;
      if (inputX < params.inputWidth && inputY < params.inputHeight) {
        best = std.max(
          best,
          poolLayout.$.src[
            hwc4Index(inputY, inputX, output.z, params.inputWidth, params.channelBlocks)
          ],
        );
      }
    }
  }

  poolLayout.$.dst[index] = maskPaddedChannels(best, output.z, params.logicalChannels);
});
