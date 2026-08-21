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

// --- GPU hand tracking kernels ---
//
// ROI preprocessor: samples a rotated square region of the camera frame
// into the hand-landmark model's [1,3,224,224] input ([0,1] RGB, hwc4).
// The ROI lives in display-crop uv space; rotation aligns the hand upright
// (wrist -> middle-MCP pointing up), which the model needs (presence drops
// from ~0.6 to ~0.03 on sideways hands).
export const RoiParams = d.struct({
  center: d.vec2f,
  size: d.vec2f,
  rotation: d.vec2f, // (cos, sin)
  cropScale: d.vec2f, // crop uv -> oriented camera uv
  cropOffset: d.vec2f,
  outputSize: d.vec2u,
  total: d.u32,
})

export const roiLayout = tgpu.bindGroupLayout({
  params: { uniform: RoiParams },
  frame: { externalTexture: d.textureExternal() },
  sampler: { sampler: 'filtering' },
  output: { storage: d.arrayOf(d.vec4f), access: 'mutable' },
})

export const roiPreprocessKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [64],
})(({ gid }) => {
  'use gpu'
  const p = roiLayout.$.params
  const index = gid.x
  if (index >= p.total) {
    return
  }
  const ox = index % p.outputSize.x
  const oy = std.intdiv(index, p.outputSize.x)
  const local = d.vec2f(
    (d.f32(ox) + 0.5) / d.f32(p.outputSize.x) - 0.5,
    (d.f32(oy) + 0.5) / d.f32(p.outputSize.y) - 0.5,
  )
  const rotated = d.vec2f(
    local.x * p.rotation.x - local.y * p.rotation.y,
    local.x * p.rotation.y + local.y * p.rotation.x,
  )
  const cropUv = d.vec2f(
    p.center.x + rotated.x * p.size.x,
    p.center.y + rotated.y * p.size.y,
  )
  const cameraUv = d.vec2f(
    p.cropOffset.x + cropUv.x * p.cropScale.x,
    p.cropOffset.y + cropUv.y * p.cropScale.y,
  )
  const rgb = std.textureSampleBaseClampToEdge(
    roiLayout.$.frame,
    roiLayout.$.sampler,
    cameraUv,
  ).rgb
  roiLayout.$.output[index] = d.vec4f(rgb, 0)
})

// Hand-depth probe v2: reads the hand-landmark OUTPUT BUFFERS directly on
// the GPU (all 21 landmarks per hand, transformed ROI -> crop space), takes
// the per-hand max disparity normalized by the GPU range. Chained after the
// landmark graphs in the same submission - no CPU sees any of it until the
// single readSync.
export const HandProbeParams = d.struct({
  outputSize: d.vec2u, // disparity grid size
  present: d.vec2u, // per-slot: hand graph ran this frame
  roi1Center: d.vec2f,
  roi1Size: d.vec2f,
  roi1Rot: d.vec2f,
  roi2Center: d.vec2f,
  roi2Size: d.vec2f,
  roi2Rot: d.vec2f,
})

export const handProbeLayout = tgpu.bindGroupLayout({
  params: { uniform: HandProbeParams },
  disparity: { storage: d.arrayOf(d.vec4f) },
  range: { storage: d.vec2f },
  lm1: { storage: d.arrayOf(d.f32) },
  lm2: { storage: d.arrayOf(d.f32) },
  result: { storage: d.vec4f, access: 'mutable' },
})

export const handDepthProbeKernel = tgpu.computeFn({ workgroupSize: [1] })(() => {
  'use gpu'
  const p = handProbeLayout.$.params
  const low = handProbeLayout.$.range.x
  const span = std.max(handProbeLayout.$.range.y - low, 0.001)
  let norm1 = d.f32(-1)
  let norm2 = d.f32(-1)
  for (let h = d.u32(0); h < 2; h++) {
    let present = p.present.x
    let center = d.vec2f(p.roi1Center)
    let size = d.vec2f(p.roi1Size)
    let rot = d.vec2f(p.roi1Rot)
    if (h !== 0) {
      present = p.present.y
      center = d.vec2f(p.roi2Center)
      size = d.vec2f(p.roi2Size)
      rot = d.vec2f(p.roi2Rot)
    }
    if (present !== 0) {
      let best = d.f32(-1)
      for (let i = d.u32(0); i < 21; i++) {
        let lxRaw = handProbeLayout.$.lm1[i * 3]!
        let lyRaw = handProbeLayout.$.lm1[i * 3 + 1]!
        if (h !== 0) {
          lxRaw = handProbeLayout.$.lm2[i * 3]!
          lyRaw = handProbeLayout.$.lm2[i * 3 + 1]!
        }
        const lx = lxRaw / 224.0 - 0.5
        const ly = lyRaw / 224.0 - 0.5
        const u = center.x + (lx * rot.x - ly * rot.y) * size.x
        const v = center.y + (lx * rot.y + ly * rot.x) * size.y
        const gx = std.clamp(u * d.f32(p.outputSize.x), 0, d.f32(p.outputSize.x - 1))
        const gy = std.clamp(v * d.f32(p.outputSize.y), 0, d.f32(p.outputSize.y - 1))
        const value =
          handProbeLayout.$.disparity[d.u32(gy) * p.outputSize.x + d.u32(gx)].x
        best = std.max(best, value)
      }
      const norm = std.clamp((best - low) / span, 0, 1)
      if (h === 0) {
        if (best >= 0) {
          norm1 = norm
        }
      } else {
        if (best >= 0) {
          norm2 = norm
        }
      }
    }
  }
  handProbeLayout.$.result = d.vec4f(norm1, norm2, low, handProbeLayout.$.range.y)
})

export interface RawDispatch {
  pipeline: GPUComputePipeline
  bindGroup: GPUBindGroup
  x: number
  y: number
  z: number
}

export interface HandGraphInstance {
  // Per-slot dedicated graph: own arena + weights (2.4MB, trivial), so two
  // hands run independently in the same submission.
  dispatches: readonly RawDispatch[]
  roiParams: TgpuUniform<typeof RoiParams>
  roiParamsRaw: GPUBuffer
  inputRaw: GPUBuffer
  outputRaw: GPUBuffer
}

export interface DepthartRunner {
  root: TgpuRoot
  depthW: number
  depthH: number
  dispatches: readonly RawDispatch[]
  preprocess: {
    pipelineRaw: GPUComputePipeline
    layoutRaw: GPUBindGroupLayout
    params: TgpuUniform<typeof FrameParams>
    paramsRaw: GPUBuffer
    samplerRaw: GPUSampler
    outputRaw: GPUBuffer
    total: number
  }
  range: readonly RawDispatch[]
  // GPU hand tracking: ROI preprocess + landmark graph per slot, plus the
  // landmark-reading depth probe. One combined staging buffer feeds the
  // single per-frame readSync: [0..511] hand1 landmarks (128 f32),
  // [512..1023] hand2, [1024..1039] probe (norm1, norm2, low, high).
  hands: {
    roiPipelineRaw: GPUComputePipeline
    roiLayoutRaw: GPUBindGroupLayout
    instances: readonly [HandGraphInstance, HandGraphInstance]
    probePipelineRaw: GPUComputePipeline
    probeParams: TgpuUniform<typeof HandProbeParams>
    probeBindGroupRaw: GPUBindGroup
    probeResultRaw: GPUBuffer
  }
  disparityRawBuffer: GPUBuffer
  rangeRawBuffer: GPUBuffer
  staging: GPUBuffer
}

const HAND_LM_BYTES = 512
export const STAGING_LAYOUT = {
  hand1: 0,
  hand2: HAND_LM_BYTES,
  probe: HAND_LM_BYTES * 2,
  total: HAND_LM_BYTES * 2 + 16,
}

async function fetchBundle(moduleId: number): Promise<ReturnType<typeof parseDepthBundle>> {
  const asset = Image.resolveAssetSource(moduleId)
  if (asset?.uri == null) throw new Error('bundle asset not resolvable')
  const response = await fetch(asset.uri)
  return parseDepthBundle(await response.arrayBuffer())
}

function loadGraph(root: TgpuRoot, bundle: ReturnType<typeof parseDepthBundle>) {
  const arena = new DepthTensorArena(root, bundle)
  const weights = createImmutableWeightStorage(
    root,
    bundle.weightSections,
    outerProductPointwiseWeights(bundle),
  )
  const prepared = createDepthDispatches(root, bundle, arena, weights)
  return { arena, prepared }
}

function unwrapDispatches(
  root: TgpuRoot,
  prepared: ReturnType<typeof createDepthDispatches>,
): RawDispatch[] {
  return prepared.dispatches.map((item) => ({
    pipeline: root.unwrap(item.pipeline),
    bindGroup: root.unwrap(item.bindGroup),
    x: item.workgroups.x,
    y: item.workgroups.y ?? 1,
    z: item.workgroups.z ?? 1,
  }))
}

export async function createDepthartRunner(device: GPUDevice): Promise<DepthartRunner> {
  const root = tgpu.initFromDevice({ device })

  const depthBundle = await fetchBundle(
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    require('../assets/depthart-relative-s-448-balanced.depthart'),
  )
  const handBundle = await fetchBundle(
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    require('../assets/hand-landmark-lite-224.depthart'),
  )

  const depthInput = depthBundle.tensorById.get(depthBundle.input.tensorId)!
  const [, , inputHeight, inputWidth] = depthInput.shape

  const depth = loadGraph(root, depthBundle)
  const hand1 = loadGraph(root, handBundle)
  const hand2 = loadGraph(root, handBundle)

  const preParams = root.createUniform(FrameParams)
  const preSampler = root.createSampler({ magFilter: 'nearest', minFilter: 'nearest' })
  const prePipeline = root.createComputePipeline({ compute: depthFramePreprocessKernel })

  const rangeBuffer = root.createBuffer(d.vec2f).$usage('storage')
  const rangeEstimator = new DepthDisparityRangeEstimator(root)
  rangeEstimator.attach(
    depth.arena.outputBuffer,
    rangeBuffer,
    inputWidth * inputHeight,
  )

  const roiPipeline = root.createComputePipeline({ compute: roiPreprocessKernel })
  const roi1Params = root.createUniform(RoiParams)
  const roi2Params = root.createUniform(RoiParams)

  const probeParams = root.createUniform(HandProbeParams)
  const probeResult = root.createBuffer(d.vec4f).$usage('storage')
  const probePipeline = root.createComputePipeline({ compute: handDepthProbeKernel })

  const graphPipelines = [
    ...new Set(
      [...depth.prepared.dispatches, ...hand1.prepared.dispatches, ...hand2.prepared.dispatches].map(
        (item) => item.pipeline,
      ),
    ),
  ]
  await Promise.all([
    prePipeline.initAsync(),
    roiPipeline.initAsync(),
    probePipeline.initAsync(),
    rangeEstimator.initAsync(),
    ...graphPipelines.map((pipeline) => pipeline.initAsync()),
  ])

  const rangeParts = rangeEstimator.parts
  const rawRange: RawDispatch[] = rangeParts.pipelines.map((pipeline, i) => ({
    pipeline: root.unwrap(pipeline),
    bindGroup: root.unwrap(rangeParts.bindGroup!),
    x: rangeParts.workgroups[i]!,
    y: 1,
    z: 1,
  }))

  // The probe bind group is fully static (no external texture): raw once.
  const probePipelineRaw = root.unwrap(probePipeline)
  const probeBindGroupRaw = device.createBindGroup({
    layout: probePipelineRaw.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: root.unwrap(probeParams.buffer) } },
      { binding: 1, resource: { buffer: depth.arena.outputBuffer.buffer } },
      { binding: 2, resource: { buffer: rangeBuffer.buffer } },
      { binding: 3, resource: { buffer: hand1.arena.outputBuffer.buffer } },
      { binding: 4, resource: { buffer: hand2.arena.outputBuffer.buffer } },
      { binding: 5, resource: { buffer: probeResult.buffer } },
    ],
  })

  return {
    root,
    depthW: inputWidth,
    depthH: inputHeight,
    dispatches: unwrapDispatches(root, depth.prepared),
    preprocess: {
      pipelineRaw: root.unwrap(prePipeline),
      layoutRaw: root.unwrap(preprocessLayout),
      params: preParams,
      paramsRaw: root.unwrap(preParams.buffer),
      samplerRaw: root.unwrap(preSampler),
      outputRaw: depth.arena.inputBuffer.buffer,
      total: inputWidth * inputHeight,
    },
    range: rawRange,
    hands: {
      roiPipelineRaw: root.unwrap(roiPipeline),
      roiLayoutRaw: root.unwrap(roiLayout),
      instances: [
        {
          dispatches: unwrapDispatches(root, hand1.prepared),
          roiParams: roi1Params,
          roiParamsRaw: root.unwrap(roi1Params.buffer),
          inputRaw: hand1.arena.inputBuffer.buffer,
          outputRaw: hand1.arena.outputBuffer.buffer,
        },
        {
          dispatches: unwrapDispatches(root, hand2.prepared),
          roiParams: roi2Params,
          roiParamsRaw: root.unwrap(roi2Params.buffer),
          inputRaw: hand2.arena.inputBuffer.buffer,
          outputRaw: hand2.arena.outputBuffer.buffer,
        },
      ],
      probePipelineRaw,
      probeParams,
      probeBindGroupRaw,
      probeResultRaw: probeResult.buffer,
    },
    disparityRawBuffer: depth.arena.outputBuffer.buffer,
    rangeRawBuffer: rangeBuffer.buffer,
    staging: device.createBuffer({
      size: STAGING_LAYOUT.total,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'hand-depth-staging',
    }),
  }
}
