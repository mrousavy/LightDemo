import { Image } from 'react-native'
// Side effect: registers TypeGPU's worklet serializers with
// react-native-worklets so roots/pipelines/bind groups/buffers can be
// captured by the frame worklet (@typegpu/react's react-native entry calls
// registerTypegpuReactSerializables at import time).
import '@typegpu/react'
import { d, std, tgpu } from 'typegpu'
import type {
  StorageFlag,
  TgpuBindGroup,
  TgpuBuffer,
  TgpuComputePipeline,
  TgpuRoot,
  TgpuSampler,
  TgpuUniform,
} from 'typegpu'
import { parseDepthBundle } from './depthart/bundle'
import { outerProductPointwiseWeights } from './depthart/conv-dispatches'
import { createDepthDispatches } from './depthart/dispatches'
import { DepthDisparityRangeEstimator } from './depthart/disparity-range'
import { createImmutableWeightStorage } from './depthart/gpu-resources'
import {
  depthFramePreprocessKernel,
  FrameParams,
  preprocessLayout,
} from './depthart/preprocess'
import { DepthTensorArena } from './depthart/tensor-arena'

// GPU hand-depth probe: samples the disparity buffer at up to 2x21 Vision
// landmarks and takes the per-hand MAX (nearest), normalized by the GPU
// 2-98% range - the depth map is never read by the CPU. The tiny result
// buffer (norm1, norm2, low, high) is read back asynchronously on the main
// JS thread at ~30Hz; the 1-frame staleness disappears inside the z EMA.
export const ProbeParams = d.struct({
  outputSize: d.vec2u,
  count1: d.u32,
  count2: d.u32,
  // 48 probe points packed two-per-vec4 (xy, zw): a vec2f array in a
  // uniform has stride 8, which is not a portable uniform layout.
  points: d.arrayOf(d.vec4f, 24),
})

export const probeLayout = tgpu.bindGroupLayout({
  params: { uniform: ProbeParams },
  disparity: { storage: d.arrayOf(d.vec4f) },
  range: { storage: d.vec2f },
  result: { storage: d.vec4f, access: 'mutable' },
})

const handMaxDisparity = (first: number, count: number) => {
  'use gpu'
  const params = probeLayout.$.params
  let best = d.f32(-1)
  for (let i = d.i32(0); i < count; i++) {
    const index = d.u32(first + i)
    const pair = params.points[index >> 1]
    const point = (index & 1) === 0 ? pair.xy : pair.zw
    const x = std.clamp(
      d.u32(point.x * d.f32(params.outputSize.x)),
      d.u32(0),
      params.outputSize.x - 1,
    )
    const y = std.clamp(
      d.u32(point.y * d.f32(params.outputSize.y)),
      d.u32(0),
      params.outputSize.y - 1,
    )
    const value = probeLayout.$.disparity[y * params.outputSize.x + x].x
    best = std.max(best, value)
  }
  return best
}

export const handProbeKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu'
  const params = probeLayout.$.params
  const low = probeLayout.$.range.x
  const span = std.max(probeLayout.$.range.y - low, 0.001)
  const best1 = handMaxDisparity(0, d.i32(params.count1))
  const best2 = handMaxDisparity(24, d.i32(params.count2))
  let norm1 = d.f32(-1)
  if (best1 >= 0) {
    norm1 = std.clamp((best1 - low) / span, 0, 1)
  }
  let norm2 = d.f32(-1)
  if (best2 >= 0) {
    norm2 = std.clamp((best2 - low) / span, 0, 1)
  }
  probeLayout.$.result = d.vec4f(norm1, norm2, low, probeLayout.$.range.y)
})

export interface DepthartRunner {
  root: TgpuRoot
  depthW: number
  depthH: number
  // Per-frame inference dispatch list (preprocess is encoded separately).
  dispatches: readonly {
    pipeline: TgpuComputePipeline
    bindGroup: TgpuBindGroup
    workgroups: { x: number; y?: number; z?: number }
  }[]
  preprocess: {
    pipeline: TgpuComputePipeline
    params: TgpuUniform<typeof FrameParams>
    sampler: TgpuSampler
    output: TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag
    layout: typeof preprocessLayout
    total: number
  }
  range: {
    pipelines: readonly TgpuComputePipeline[]
    bindGroup: TgpuBindGroup
    workgroups: readonly number[]
  }
  probe: {
    pipeline: TgpuComputePipeline
    params: TgpuUniform<typeof ProbeParams>
    bindGroup: TgpuBindGroup
    result: TgpuBuffer<d.Vec4f> & StorageFlag
  }
  // Raw GPUBuffers for the raw-WebGPU lighting passes.
  disparityRawBuffer: GPUBuffer
  rangeRawBuffer: GPUBuffer
  probeResultRaw: GPUBuffer
  // MAP_READ staging for the same-frame readSync of the probe result.
  probeStaging: GPUBuffer
}

export async function createDepthartRunner(device: GPUDevice): Promise<DepthartRunner> {
  const root = tgpu.initFromDevice({ device })

  const asset = Image.resolveAssetSource(
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    require('../assets/depthart-relative-s-448-balanced.depthart'),
  )
  if (asset?.uri == null) throw new Error('DepthART bundle asset not resolvable')
  const response = await fetch(asset.uri)
  const bytes = await response.arrayBuffer()
  const bundle = parseDepthBundle(bytes)

  const inputTensor = bundle.tensorById.get(bundle.input.tensorId)!
  const outputTensor = bundle.tensorById.get(bundle.output.tensorId)!
  const [, , inputHeight, inputWidth] = inputTensor.shape
  const [, , outputHeight, outputWidth] = outputTensor.shape

  const arena = new DepthTensorArena(root, bundle)
  const weights = createImmutableWeightStorage(
    root,
    bundle.weightSections,
    outerProductPointwiseWeights(bundle),
  )
  const prepared = createDepthDispatches(root, bundle, arena, weights)

  const preParams = root.createUniform(FrameParams)
  const preSampler = root.createSampler({ magFilter: 'nearest', minFilter: 'nearest' })
  const prePipeline = root.createComputePipeline({ compute: depthFramePreprocessKernel })

  const rangeBuffer = root.createBuffer(d.vec2f).$usage('storage')
  const rangeEstimator = new DepthDisparityRangeEstimator(root)
  rangeEstimator.attach(arena.outputBuffer, rangeBuffer, outputWidth * outputHeight)

  const probeParams = root.createUniform(ProbeParams)
  const probeResult = root.createBuffer(d.vec4f).$usage('storage')
  const probePipeline = root.createComputePipeline({ compute: handProbeKernel })
  const probeBindGroup = root.createBindGroup(probeLayout, {
    params: probeParams,
    disparity: arena.outputBuffer,
    range: rangeBuffer,
    result: probeResult,
  })

  const uniquePipelines = [...new Set(prepared.dispatches.map((item) => item.pipeline))]
  await Promise.all([
    prePipeline.initAsync(),
    probePipeline.initAsync(),
    rangeEstimator.initAsync(),
    ...uniquePipelines.map((pipeline) => pipeline.initAsync()),
  ])

  return {
    root,
    depthW: inputWidth,
    depthH: inputHeight,
    dispatches: prepared.dispatches as DepthartRunner['dispatches'],
    preprocess: {
      pipeline: prePipeline,
      params: preParams,
      sampler: preSampler,
      output: arena.inputBuffer as unknown as DepthartRunner['preprocess']['output'],
      layout: preprocessLayout,
      total: inputWidth * inputHeight,
    },
    range: rangeEstimator.parts as DepthartRunner['range'],
    probe: {
      pipeline: probePipeline,
      params: probeParams,
      bindGroup: probeBindGroup,
      result: probeResult as DepthartRunner['probe']['result'],
    },
    disparityRawBuffer: arena.outputBuffer.buffer,
    rangeRawBuffer: rangeBuffer.buffer,
    probeResultRaw: probeResult.buffer,
    probeStaging: device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'probe-staging',
    }),
  }
}
