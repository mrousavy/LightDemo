import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  ActivityIndicator,
  Linking,
  PixelRatio,
  Pressable,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native'
import {
  Canvas,
  useCanvasRef,
  type NativeCanvas,
  type RNCanvasContext,
} from 'react-native-webgpu'
import {
  useCamera,
  useCameraDevices,
  useCameraPermission,
  useFrameOutput,
  type Frame,
} from 'react-native-vision-camera'
import {
  LightNativeModule,
  type LightControls,
  type LightPipeline,
  type LightStatus,
} from 'light-native'
import { d } from 'typegpu'
import { createSynchronizable } from 'react-native-worklets'
import { createDepthartRunner, type DepthartRunner } from './depthartRunner'
import { DEPTH_PREPARE_SHADER, RELIGHT_SHADER, SURFACE_SHADER } from './shaders'

const REQUIRED_FEATURES: GPUFeatureName[] = [
  'rnwebgpu/native-texture' as GPUFeatureName,
  'dawn-multi-planar-formats' as GPUFeatureName,
  // DepthART's fp16 'balanced' weight bundle runs in f16 compute.
  'shader-f16' as GPUFeatureName,
  // The frame worklet and the main thread (hand-probe readback) both submit
  // to the device; this makes Dawn mutex it internally.
  'implicit-device-synchronization' as GPUFeatureName,
]

// Unified z-space, MUST match SURFACE_FAR_Z in shaders.ts: normalized
// disparity 1 (nearest object) -> z = 0, disparity 0 (farthest) -> -Z_FAR.
// Positive z is toward the viewer, in front of the entire scene.
const Z_FAR = 1.1
const LIGHT_Z_MIN = -Z_FAR
const LIGHT_Z_MAX = 0.6
// The light rides this far IN FRONT of the depth-map plane sampled at the
// controlling fingertips - in front of the hand by construction. Also
// larger than BULB_WORLD_RADIUS (0.05) so the sphere front never clips.
const Z_MARGIN = 0.1

const DEFAULT_CONTROLS: LightControls = {
  mirror: true,
  // Hardcoded upright rotation for the Insta360 Link 2 Pro's current
  // mounting (verified visually). Cycle the Rot button if it changes;
  // 'auto' falls back to VisionCamera's tag, which is wrong for external
  // cameras (derived from the connection's default portrait orientation).
  rotationOverride: 270,
  mode: 0,
  // Moodier than the TypeGPU defaults (intensity 3.0 / exposure 0.5): a dim
  // base scene with a bright light reads far more dramatic on a well-lit
  // webcam feed.
  intensity: 4.5,
  exposure: 0.22,
  relief: 0.85,
  specular: 0.28,
  shadow: 0.85,
  occlusion: 0.7,
  colorR: 1.0,
  colorG: 0.83,
  colorB: 0.6,
  touchX: 0.34,
  touchY: 0.34,
  touchActive: false,
  lightZ: 0.25,
  handControl: true,
  snapshotPath: '',
}

const MODE_NAMES = ['Relit', 'Camera', 'Depth', 'Normals']

// The Insta360's sensor aspect: what the display shows and the depth model
// covers (centered crop of the oriented frame).
const DISPLAY_ASPECT = 4 / 3

// Stable identities - inline literals would recreate the frame output /
// reconfigure the session on every React render.
// The Insta360's sensor is native 4:3 (up to 3840x2880); its 16:9 modes are
// crops. 1280x960 uses the full sensor height - more vertical FOV for our
// square center crop.
const TARGET_RESOLUTION = { width: 1920, height: 1440 }
// The synchronous pipeline is model-paced at ~30fps; streaming the camera
// at 60fps only burns ISP/memory bandwidth on frames we drop.
const CAMERA_CONSTRAINTS = [{ fps: 30 }]

interface PipelineState {
  context: RNCanvasContext
  sampler: GPUSampler
  depthPreparePipeline: GPUComputePipeline
  surfacePipeline: GPUComputePipeline
  relightPipeline: GPURenderPipeline
  computeParamsBuffer: GPUBuffer
  relightParamsBuffer: GPUBuffer
  historyBuffer: GPUBuffer
  surfaceBindGroup: GPUBindGroup
  surfaceView: GPUTextureView
  depthW: number
  depthH: number
  // Lighting-field resolution: FIELD_SCALE x the model output (see shaders).
  fieldW: number
  fieldH: number
}

export function LightScreen() {
  const { hasPermission, requestPermission } = useCameraPermission()
  useEffect(() => {
    if (!hasPermission) requestPermission()
  }, [hasPermission, requestPermission])

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text style={styles.info}>Camera permission required.</Text>
        <Pressable style={styles.button} onPress={() => Linking.openSettings()}>
          <Text style={styles.buttonText}>Open Settings</Text>
        </Pressable>
      </View>
    )
  }
  return <LightView />
}

function LightView() {
  const ref = useCanvasRef()
  const window = useWindowDimensions()
  const [device, setDevice] = useState<GPUDevice | null>(null)
  const [depthart, setDepthart] = useState<DepthartRunner | null>(null)
  const [nitro, setNitro] = useState<LightPipeline | null>(null)
  const [pipeline, setPipeline] = useState<PipelineState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<LightStatus | null>(null)
  const [controls, setControls] = useState<LightControls>(DEFAULT_CONTROLS)

  // Capture the RNWebGPU singleton so the worklet can call the interop
  // factory (it crosses the worklet boundary via the WebGPU serializer).
  const rnwgpu = RNWebGPU

  // 1. GPU device
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const adapter = await navigator.gpu.requestAdapter()
        if (adapter == null) throw new Error('requestAdapter returned null')
        const gpuDevice = await adapter.requestDevice({
          requiredFeatures: REQUIRED_FEATURES,
        })
        if (!cancelled) setDevice(gpuDevice)
      } catch (e) {
        if (!cancelled) setError(`GPU device: ${String(e)}`)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  // 2. Native pipeline (CoreML model load + warmup)
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const start = performance.now()
        const p = await LightNativeModule.createPipeline()
        console.log(
          `[LightDemo] pipeline ready in ${(performance.now() - start).toFixed(0)}ms, ` +
            `depth ${p.depthWidth}x${p.depthHeight}`,
        )
        if (!cancelled) setNitro(p)
      } catch (e) {
        if (!cancelled) setError(`CoreML pipeline: ${String(e)}`)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  // 2b. DepthART GPU inference (TypeGPU on the same Dawn device): the
  // entire depth model runs as ~250 compute dispatches inside our frame
  // encoder - no CoreML, no ANE, no CPU copies.
  useEffect(() => {
    if (device == null || depthart != null) return
    let cancelled = false
    ;(async () => {
      try {
        const start = performance.now()
        const runner = await createDepthartRunner(device)
        console.log(
          `[LightDemo] depthart ready in ${(performance.now() - start).toFixed(0)}ms, ` +
            `${runner.depthW}x${runner.depthH}, ${runner.dispatches.length} dispatches`,
        )
        if (!cancelled) setDepthart(runner)
      } catch (e) {
        console.log(`[LightDemo] depthart init FAILED: ${String(e)}`)
        if (!cancelled) setError(`DepthART init: ${String(e)}`)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [device, depthart])

  // 3. WebGPU pipelines (device + canvas + depth size ready)
  useEffect(() => {
    if (device == null || nitro == null || depthart == null || pipeline != null) return
    const missing = REQUIRED_FEATURES.filter((f) => !device.features.has(f))
    if (missing.length > 0) {
      setError(`Device missing features: ${missing.join(', ')}`)
      return
    }
    const context = ref.current?.getContext('webgpu')
    if (context == null) return
    const canvas = context.canvas as unknown as NativeCanvas
    canvas.width = canvas.clientWidth * PixelRatio.get()
    canvas.height = canvas.clientHeight * PixelRatio.get()
    const format = navigator.gpu.getPreferredCanvasFormat()
    context.configure({ device, format, alphaMode: 'opaque' })

    const depthW = depthart.depthW
    const depthH = depthart.depthH
    // The disparity grid is a 448x448 anamorphic squeeze of the full 4:3
    // frame; the lighting field keeps the DISPLAY aspect so field texels
    // stay square on screen.
    const fieldW = 1344
    const fieldH = 1008
    const fieldTexelCount = fieldW * fieldH

    const depthPrepareModule = device.createShaderModule({ code: DEPTH_PREPARE_SHADER })
    const surfaceModule = device.createShaderModule({ code: SURFACE_SHADER })
    const relightModule = device.createShaderModule({ code: RELIGHT_SHADER })

    const depthPreparePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: depthPrepareModule, entryPoint: 'depthPrepare' },
    })
    const surfacePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: surfaceModule, entryPoint: 'surfacePass' },
    })
    const relightPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: relightModule, entryPoint: 'vs_main' },
      fragment: { module: relightModule, entryPoint: 'fs_main', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    })

    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' })
    const computeParamsBuffer = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const relightParamsBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const historyBuffer = device.createBuffer({
      size: fieldTexelCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const surfaceTexture = device.createTexture({
      size: [fieldW, fieldH],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    const surfaceView = surfaceTexture.createView()

    const surfaceBindGroup = device.createBindGroup({
      layout: surfacePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: computeParamsBuffer } },
        { binding: 1, resource: { buffer: historyBuffer } },
        { binding: 2, resource: surfaceView },
      ],
    })

    setPipeline({
      context,
      sampler,
      depthPreparePipeline,
      surfacePipeline,
      relightPipeline,
      computeParamsBuffer,
      relightParamsBuffer,
      historyBuffer,
      surfaceBindGroup,
      surfaceView,
      depthW,
      depthH,
      fieldW,
      fieldH,
    })
  }, [device, nitro, depthart, pipeline, ref])

  // 4. Push UI controls into the native store (worklet reads them per frame)
  useEffect(() => {
    nitro?.setControls(controls)
  }, [nitro, controls])

  // Debug handles for the CDP console (scripts/jsconsole.mjs).
  useEffect(() => {
    ;(globalThis as Record<string, unknown>).__nitro = nitro
    ;(globalThis as Record<string, unknown>).__nitroFactory = LightNativeModule
  }, [nitro])

  // 5. Poll status for the HUD
  useEffect(() => {
    if (nitro == null) return
    const interval = setInterval(() => setStatus(nitro.getStatus()), 500)
    return () => clearInterval(interval)
  }, [nitro])

  // 6. Watchdog: external UVC cameras (the Insta360) can silently stall
  // their stream - no AVFoundation interruption, error, or stop event fires
  // and the session still reports running (verified via thread sample: the
  // frame thread just idles waiting for work). Detect a frozen frameCount
  // and bounce the session via isActive.
  //
  // Gentle by design: external cameras can take several seconds to deliver
  // their first frame after a session (re)start, so each restart gets a
  // 12s grace period, and repeated restarts back off exponentially -
  // aggressive stop/start churn can prevent a recovering camera from ever
  // coming up (and stresses flaky UVC drivers).
  const [cameraSuspended, setCameraSuspended] = useState(false)
  const watchdog = useMemo(
    () => ({
      lastCount: -1,
      stalledMs: 0,
      graceUntil: Date.now() + 15000,
      backoffMs: 5000,
    }),
    [],
  )
  useEffect(() => {
    if (nitro == null || pipeline == null) return
    const interval = setInterval(() => {
      const count = nitro.getStatus().frameCount
      if (count !== watchdog.lastCount) {
        // Frames are flowing - reset stall tracking and backoff.
        watchdog.stalledMs = 0
        watchdog.backoffMs = 5000
      } else if (!cameraSuspended && Date.now() > watchdog.graceUntil) {
        watchdog.stalledMs += 2000
        if (watchdog.stalledMs >= watchdog.backoffMs) {
          console.log(
            `[LightDemo] camera stall (frameCount frozen at ${count} for ` +
              `${watchdog.stalledMs / 1000}s) - restarting session ` +
              `(next attempt in ${(watchdog.backoffMs * 2) / 1000}s)`,
          )
          watchdog.stalledMs = 0
          watchdog.backoffMs = Math.min(watchdog.backoffMs * 2, 40000)
          watchdog.graceUntil = Date.now() + 12000
          setCameraSuspended(true)
          setTimeout(() => setCameraSuspended(false), 600)
        }
      }
      watchdog.lastCount = count
    }, 2000)
    return () => clearInterval(interval)
  }, [nitro, pipeline, cameraSuspended, watchdog])

  // 7. Debug: periodic window snapshot into Documents/snap.png so rendering
  // can be verified headlessly from outside the app.
  useEffect(() => {
    if (!__DEV__) return
    const interval = setInterval(() => {
      LightNativeModule.snapshotWindow('snap.png').catch((e) =>
        console.log(`[LightDemo] snapshot failed: ${String(e)}`),
      )
    }, 3000)
    return () => clearInterval(interval)
  }, [])

  const devices = useCameraDevices()
  // Never pick a Continuity Camera (iPhone). Prefer USB/external cameras,
  // then the built-in front camera. A crashed UVC driver can leave a
  // phantom "NULL Camera" device behind that fails every session config
  // with AVFoundation -11800 '!obj' - never select it.
  const cameraDevice = useMemo(() => {
    const real = devices.filter(
      (d) =>
        !d.isContinuityCamera &&
        d.type !== 'continuity' &&
        !d.localizedName.includes('NULL'),
    )
    const device =
      real.find((d) => d.type === 'external') ??
      real.find((d) => d.position === 'front') ??
      real.find((d) => d.position === 'back') ??
      real[0]
    if (device != null) {
      console.log(
        `[LightDemo] camera: "${device.localizedName}" type=${device.type} ` +
          `position=${device.position} (${devices.length} devices total)`,
      )
    }
    return device
  }, [devices])


  // Worklet-persistent mutable state. Captured once - the worklet closure is
  // only recreated if one of the (stable-after-init) captured values changes.
  const box = useMemo(
    () => ({
      frameCount: 0,
      lastFrameTime: 0,
      fps: 0,
      lastDepthSeq: -1,
      lastHandSeq: -1,
      rangeLow: 0,
      rangeHigh: 1,
      rangeInitialized: false,
      lightX: 0.34,
      lightY: 0.34,
      lightZ: 0.25,
      bulbScale: 1,
      grabbedHandSize: 0,
      grabbed: false,
      pinchFrames: 0,
      releaseFrames: 0,
      everControlled: false,
      loggedFirstFrame: false,
      handX: 0,
      handY: 0,
      handValid: false,
      outlierCount: 0,
      pinchSmoothed: 1,
      velX: 0,
      velY: 0,
      lostFrames: 0,
      depthEverRun: false,
      tSync: 0,
      tEnc: 0,
      tSub: 0,
      tWait: 0,
      // Reused every frame (48 [x, y] slots) - per-frame allocation here
      // showed up as rare GC stutters.
      probePoints: Array.from({ length: 48 }, () => [0, 0]) as number[][],
    }),
    [],
  )

  // GPU hand-depth probe results, read back asynchronously on the main JS
  // thread and shared into the frame worklet (1-frame staleness is
  // invisible after the z EMA).
  const gpuDepth = useMemo(
    () => createSynchronizable({ h1: -1, h2: -1, low: 0, high: 1 }),
    [],
  )
  useEffect(() => {
    if (depthart == null) return
    let live = true
    let busy = false
    const interval = setInterval(() => {
      if (busy) return
      busy = true
      ;(depthart.probe.result as { read(): Promise<{ x: number; y: number; z: number; w: number }> })
        .read()
        .then((v) => {
          if (live) gpuDepth.setBlocking({ h1: v.x, h2: v.y, low: v.z, high: v.w })
          ;(globalThis as Record<string, unknown>).__gpuDepth = { h1: v.x, h2: v.y, low: v.z, high: v.w }
        })
        .catch(() => {})
        .finally(() => {
          busy = false
        })
    }, 33)
    // Debug: dump the model INPUT tensor (what DepthART actually sees).
    ;(globalThis as Record<string, unknown>).__dumpInput = async (outPath: string) => {
      const W = depthart.depthW
      const H = depthart.depthH
      const input = await (
        depthart.preprocess.output as unknown as {
          read(): Promise<{ x: number; y: number; z: number }[]>
        }
      ).read()
      const rgba = new Uint8Array(W * H * 4)
      for (let i = 0; i < W * H; i++) {
        const v = input[i]!
        rgba[i * 4] = Math.max(0, Math.min(255, (v.x * 0.229 + 0.485) * 255))
        rgba[i * 4 + 1] = Math.max(0, Math.min(255, (v.y * 0.224 + 0.456) * 255))
        rgba[i * 4 + 2] = Math.max(0, Math.min(255, (v.z * 0.225 + 0.406) * 255))
        rgba[i * 4 + 3] = 255
      }
      await LightNativeModule.savePng(outPath, W, H, W * 4, false, rgba.buffer as ArrayBuffer)
      return 'saved'
    }
    return () => {
      live = false
      clearInterval(interval)
    }
  }, [depthart, gpuDepth])

  // Stable worklet: identity must only change when the captured pipeline
  // objects change, otherwise every React render re-serializes the closure
  // and resets the worklet-side `box` state.
  const onFrame = useCallback(
    (frame: Frame) => {
      'worklet'
      if (pipeline == null || device == null || nitro == null || depthart == null) {
        frame.dispose()
        return
      }
      const renderStart = performance.now()
      if (!box.loggedFirstFrame) {
        box.loggedFirstFrame = true
        console.log(
          `[LightDemo] first frame: ${frame.width}x${frame.height} ` +
            `${frame.pixelFormat} orientation=${frame.orientation} ` +
            `mirrored=${frame.isMirrored} hasNativeBuffer=${frame.hasNativeBuffer}`,
        )
      }
      const controlsNow = nitro.getControls()
      // Rotation needed to display the buffer upright. Applied identically to
      // the camera texture (Dawn), the depth-model input (CoreImage) and the
      // hand detector (Vision) so all three stay in the same display space.
      // The default comes from DEFAULT_CONTROLS (hardcoded for this camera's
      // mounting - VisionCamera's orientation tag is wrong for external
      // cameras); cycle the Rot button if the gimbal changes. 'auto' (-1)
      // falls back to the tag.
      let rotationDeg: 0 | 90 | 180 | 270 = 0
      if (controlsNow.rotationOverride >= 0) {
        rotationDeg = controlsNow.rotationOverride as 0 | 90 | 180 | 270
      } else if (frame.orientation === 'right') rotationDeg = 90
      else if (frame.orientation === 'down') rotationDeg = 180
      else if (frame.orientation === 'left') rotationDeg = 270
      const nativeBuffer = frame.getNativeBuffer()
      try {

        const videoFrame = rnwgpu.createVideoFrameFromNativeBuffer(nativeBuffer.pointer)
        try {
          // XOR: mirror the display when exactly one of (buffer mirrored,
          // user wants mirror) holds. Applied consistently to the camera
          // fetch, the surface texture, and the hand coordinates.
          const mirrored = frame.isMirrored !== controlsNow.mirror
          const rotated = rotationDeg === 90 || rotationDeg === 270
          const dispW = rotated ? videoFrame.height : videoFrame.width
          const dispH = rotated ? videoFrame.width : videoFrame.height

          // The oriented camera frame is PORTRAIT 3:4 (the Insta360 delivers
          // a rotated buffer); display and depth both use its centered 4:3
          // landscape crop - the same region the old CoreML prep extracted.
          const frameAspect = dispW / dispH
          let cropW = 1
          let cropH = 1
          if (frameAspect > DISPLAY_ASPECT) {
            cropW = DISPLAY_ASPECT / frameAspect
          } else {
            cropH = frameAspect / DISPLAY_ASPECT
          }
          const cropOffX = (1 - cropW) / 2
          const cropOffY = (1 - cropH) / 2

          // Imported once per frame (external textures expire each frame),
          // used by BOTH the JBU guide fetch and the relight pass.
          const externalTexture = device.importExternalTexture({
            source: videoFrame,
            label: 'camera-frame',
            rotation: rotationDeg,
          })

          // --- hands: native Vision on the prepared buffer (its 4:3 crop
          // of the 4:3 camera is the identity, so landmarks are full-frame
          // uv - the same space as the GPU disparity grid). Depth itself no
          // longer runs natively. ---
          const tSync0 = performance.now()
          const depth = nitro.analyzeSync(frame, rotationDeg, controlsNow.handControl, false)
          const tSync = performance.now() - tSync0
          const tEnc0 = performance.now()

          // --- DepthART inference: a typed TypeGPU encoder wraps the whole
          // frame; the raw lighting passes below record into the SAME
          // encoder via root.unwrap, one submission for everything. ---
          const tgpuEncoder = depthart.root['~unstable'].createCommandEncoder()
          const infPass = tgpuEncoder.beginComputePass()

          // 1. camera -> normalized [1,3,448,448] input tensor. The external
          // texture import already BAKES the display rotation into uv space
          // (that is what keeps the camera view upright), so the transform
          // must NOT rotate again - it is a pure center-crop scale that
          // selects the same 4:3 region the display shows, squeezed
          // anamorphically into the square model input.
          depthart.preprocess.params.write({
            uvTransform: d.mat2x2f(cropW, 0, 0, cropH),
            outputSize: d.vec2u(depthart.depthW, depthart.depthH),
            mirrorX: 0,
            swapAxes: rotated ? 1 : 0,
            total: depthart.preprocess.total,
          })
          const preBindGroup = depthart.root.createBindGroup(depthart.preprocess.layout, {
            params: depthart.preprocess.params,
            frame: externalTexture,
            sampler: depthart.preprocess.sampler,
            output: depthart.preprocess.output,
          })
          depthart.preprocess.pipeline
            .with(infPass)
            .with(preBindGroup)
            .dispatchWorkgroups(Math.ceil(depthart.preprocess.total / 64))

          // 2. the model: ~230 prepared dispatches, all inside this pass.
          for (const item of depthart.dispatches) {
            item.pipeline
              .with(infPass)
              .with(item.bindGroup)
              .dispatchWorkgroups(
                item.workgroups.x,
                item.workgroups.y ?? 1,
                item.workgroups.z ?? 1,
              )
          }

          // 3. GPU 2-98% disparity range (histogram; writes the vec2f the
          // JBU normalization reads - no CPU round-trip).
          for (let i = 0; i < depthart.range.pipelines.length; i++) {
            depthart.range.pipelines[i]
              .with(infPass)
              .with(depthart.range.bindGroup)
              .dispatchWorkgroups(depthart.range.workgroups[i])
          }

          // Join Vision: it ran on the ANE while we encoded ~260 dispatches,
          // so this wait is (17ms - encode time), not 17ms - and the
          // landmarks are SAME-FRAME.
          const tWait0 = performance.now()
          nitro.waitForHands()
          box.tWait = box.tWait * 0.9 + (performance.now() - tWait0) * 0.1
          const hand = nitro.getHandResult()

          // 4. hand-depth probe: 5-tap crosses at thumb/index/pinch of each
          // hand, max disparity normalized on GPU; read back async on the
          // main thread for the JS z state machine.
          const probePoints = box.probePoints
          const PROBE_OFF = 2 / depthart.depthW
          const PROBE_TAPS = [
            [0, 0],
            [-PROBE_OFF, 0],
            [PROBE_OFF, 0],
            [0, -PROBE_OFF],
            [0, PROBE_OFF],
          ]
          let count1 = 0
          if (hand.hand1.tracked) {
            const anchors = [
              [hand.hand1.thumbX, hand.hand1.thumbY],
              [hand.hand1.indexX, hand.hand1.indexY],
              [hand.hand1.midX, hand.hand1.midY],
            ]
            for (const a of anchors) {
              for (const t of PROBE_TAPS) {
                const slot = probePoints[count1]!
                slot[0] = a[0]! + t[0]!
                slot[1] = a[1]! + t[1]!
                count1++
              }
            }
          }
          let count2 = 0
          if (hand.hand2.tracked) {
            const anchors = [
              [hand.hand2.thumbX, hand.hand2.thumbY],
              [hand.hand2.indexX, hand.hand2.indexY],
              [hand.hand2.midX, hand.hand2.midY],
            ]
            for (const a of anchors) {
              for (const t of PROBE_TAPS) {
                const slot = probePoints[24 + count2]!
                slot[0] = a[0]! + t[0]!
                slot[1] = a[1]! + t[1]!
                count2++
              }
            }
          }
          depthart.probe.params.write({
            outputSize: d.vec2u(depthart.depthW, depthart.depthH),
            count1,
            count2,
            // Plain [x, y] tuples are a documented fast-path write form;
            // the TS type only advertises typed vectors.
            points: probePoints as unknown as d.v2f[],
          })
          depthart.probe.pipeline
            .with(infPass)
            .with(depthart.probe.bindGroup)
            .dispatchWorkgroups(1)
          infPass.end()
          const tEnc = performance.now() - tEnc0

          // Everything below records RAW WebGPU into the same encoder.
          const encoder = depthart.root.unwrap(tgpuEncoder) as GPUCommandEncoder

          // --- JBU + lighting field, every frame ---
          const reset = box.depthEverRun ? 0 : 1
          box.depthEverRun = true
          const computeParams = new ArrayBuffer(48)
          const cpU32 = new Uint32Array(computeParams)
          const cpF32 = new Float32Array(computeParams)
          cpU32[0] = pipeline.fieldW
          cpU32[1] = pipeline.fieldH
          cpU32[2] = reset
          cpU32[3] = mirrored ? 1 : 0
          cpF32[4] = depthart.depthW
          cpF32[5] = depthart.depthH
          cpF32[6] = cropW
          cpF32[7] = cropH
          cpF32[8] = cropOffX
          cpF32[9] = cropOffY
          device.queue.writeBuffer(pipeline.computeParamsBuffer, 0, computeParams)

          const depthBindGroup = device.createBindGroup({
            layout: pipeline.depthPreparePipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: pipeline.computeParamsBuffer } },
              { binding: 1, resource: { buffer: depthart.disparityRawBuffer } },
              { binding: 2, resource: { buffer: pipeline.historyBuffer } },
              { binding: 3, resource: pipeline.sampler },
              { binding: 4, resource: externalTexture },
              { binding: 5, resource: { buffer: depthart.rangeRawBuffer } },
            ],
          })
          const compute = encoder.beginComputePass()
          compute.setPipeline(pipeline.depthPreparePipeline)
          compute.setBindGroup(0, depthBindGroup)
          compute.dispatchWorkgroups(Math.ceil((pipeline.fieldW * pipeline.fieldH) / 64))
          compute.setPipeline(pipeline.surfacePipeline)
          compute.setBindGroup(0, pipeline.surfaceBindGroup)
          compute.dispatchWorkgroups(
            Math.ceil(pipeline.fieldW / 8),
            Math.ceil(pipeline.fieldH / 8),
          )
          compute.end()

          // --- hand interaction ---
          const prevLightX = box.lightX
          const prevLightY = box.lightY
          let freshHandUpdate = false
          if (controlsNow.handControl && hand.seq >= 0 && hand.seq !== box.lastHandSeq) {
            box.lastHandSeq = hand.seq
            // Up to two hands; slot order is unstable across frames, so
            // continuity is matched by proximity, never by slot index.
            const hands = []
            if (hand.hand1.tracked && hand.hand1.confidence > 0.45) hands.push(hand.hand1)
            if (hand.hand2.tracked && hand.hand2.confidence > 0.45) hands.push(hand.hand2)
            if (hands.length > 0) {
              // Classify once in display space. Semantics (per UX spec):
              // - a pinched hand ALWAYS wins (no proximity requirement)
              // - no pinch: drift to the hand, or the center between both
              // - both pinched: the one nearer the light
              // - continuity of a grab is anchored to the LOCKED hand's last
              //   position, never to the light (at the two-hand center,
              //   nearest-to-light is a coin flip onto the open hand, which
              //   caused a grab/release/jump loop)
              const detail = []
              for (const h of hands) {
                detail.push({
                  h: h,
                  x: mirrored ? 1 - h.midX : h.midX,
                  y: h.midY,
                  pinched: h.pinchRatio < 0.28,
                  open: h.pinchRatio > 0.42,
                })
              }
              const pinched = detail.filter((d) => d.pinched)

              if (!box.grabbed) {
                if (pinched.length > 0) {
                  // A pinch anywhere claims the light.
                  box.pinchFrames += 1
                  if (box.pinchFrames >= 2) {
                    let target = pinched[0]
                    let best = Number.POSITIVE_INFINITY
                    for (const d of pinched) {
                      const dist = (d.x - box.lightX) ** 2 + (d.y - box.lightY) ** 2
                      if (dist < best) {
                        best = dist
                        target = d
                      }
                    }
                    box.grabbed = true
                    box.everControlled = true
                    box.releaseFrames = 0
                    box.handX = target.x
                    box.handY = target.y
                    box.handValid = true
                    box.outlierCount = 0
                  }
                } else {
                  box.pinchFrames = 0
                  // Hover-steering: drift toward the hand(s) center - and
                  // in DEPTH too, toward the nearer followed hand's plane.
                  if (!controlsNow.touchActive) {
                    let cx = 0
                    let cy = 0
                    for (const d of detail) {
                      cx += d.x
                      cy += d.y
                    }
                    cx /= detail.length
                    cy /= detail.length
                    box.everControlled = true
                    box.lightX += (cx - box.lightX) * 0.06
                    box.lightY += (cy - box.lightY) * 0.06
                    freshHandUpdate = true
                    // GPU probe result: already range-normalized (1-frame
                    // stale via the async readback; invisible after EMA).
                    const gd = gpuDepth.getDirty()
                    let hoverNorm = -1
                    for (const item of detail) {
                      const norm = item.h === hand.hand1 ? gd.h1 : gd.h2
                      if (norm > hoverNorm) hoverNorm = norm
                    }
                    if (hoverNorm >= 0) {
                      const hoverZ = (hoverNorm - 1) * Z_FAR + Z_MARGIN
                      box.lightZ += (hoverZ - box.lightZ) * 0.08
                    }
                  }
                }
              }
              if (box.grabbed) {
                // Continuity: the locked hand is the one nearest the last
                // locked position...
                let locked = detail[0]
                let bestDistance = Number.POSITIVE_INFINITY
                for (const d of detail) {
                  const dist = (d.x - box.handX) ** 2 + (d.y - box.handY) ** 2
                  if (dist < bestDistance) {
                    bestDistance = dist
                    locked = d
                  }
                }
                // ...unless it opened while another hand is pinched: the
                // pinched hand takes the grab over (pinched wins).
                if (locked.open && pinched.length > 0 && pinched.indexOf(locked) < 0) {
                  locked = pinched[0]
                  box.handX = locked.x
                  box.handY = locked.y
                }
                const rawX = locked.x
                const rawY = locked.y
                // Reject single-frame teleports of the locked hand (Vision
                // jitters under motion blur).
                const jump = box.handValid
                  ? Math.hypot(rawX - box.handX, rawY - box.handY)
                  : 0
                const isOutlier = box.handValid && jump > 0.3 && box.outlierCount < 3
                if (isOutlier) {
                  box.outlierCount += 1
                } else {
                  box.outlierCount = 0
                  if (box.handValid) {
                    box.handX += (rawX - box.handX) * 0.6
                    box.handY += (rawY - box.handY) * 0.6
                  } else {
                    box.handX = rawX
                    box.handY = rawY
                    box.handValid = true
                  }
                  if (locked.open) {
                    box.releaseFrames += 1
                    if (box.releaseFrames >= 2) {
                      box.grabbed = false
                      box.pinchFrames = 0
                    }
                  } else {
                    box.releaseFrames = 0
                  }
                }
                if (box.grabbed && !isOutlier) {
                  // Smoothly follow the locked pinch point.
                  box.lightX += (box.handX - box.lightX) * 0.5
                  box.lightY += (box.handY - box.lightY) * 0.5
                  freshHandUpdate = true
                  // Light depth = the depth map itself: the fingertips'
                  // disparity (nearest over small tap neighborhoods at
                  // thumb, index and knuckle) mapped into the unified
                  // z-space, plus a fixed forward margin - the bulb sits
                  // just IN FRONT of the fingers by construction, in the
                  // exact same space the shading, occlusion and shadow
                  // march use. No hand-size heuristics, no envelope: where
                  // the depth map says the fingers are is where the light
                  // goes.
                  if (locked.h.handSize > 0) {
                    box.grabbedHandSize = locked.h.handSize
                  }
                  const gd = gpuDepth.getDirty()
                  const nearestNorm = locked.h === hand.hand1 ? gd.h1 : gd.h2
                  if (nearestNorm >= 0) {
                    const targetZ = (nearestNorm - 1) * Z_FAR + Z_MARGIN
                    box.lightZ += (targetZ - box.lightZ) * 0.35
                  }
                }
              }
            } else {
              box.pinchFrames = 0
              box.handValid = false
              box.outlierCount = 0
            }
          }

          // --- momentum ---
          // While the hand actively drives the light, measure its velocity;
          // whenever the driving signal disappears (fingers released, or
          // tracking lost - e.g. the hand moved behind a chair), coast on
          // that velocity and bleed it off smoothly instead of freezing.
          if (freshHandUpdate) {
            box.velX = box.velX * 0.6 + (box.lightX - prevLightX) * 0.4
            box.velY = box.velY * 0.6 + (box.lightY - prevLightY) * 0.4
            box.lostFrames = 0
          } else if (!controlsNow.touchActive && box.everControlled) {
            box.lightX += box.velX
            box.lightY += box.velY
            box.velX *= 0.93
            box.velY *= 0.93
            if (Math.abs(box.velX) < 0.0004) box.velX = 0
            if (Math.abs(box.velY) < 0.0004) box.velY = 0
            box.lightX = Math.min(Math.max(box.lightX, 0.02), 0.98)
            box.lightY = Math.min(Math.max(box.lightY, 0.02), 0.98)
            // Auto-release a grab whose hand has been gone for ~1.5s so
            // hover-steering can take over again when the hand returns.
            if (box.grabbed) {
              box.lostFrames += 1
              if (box.lostFrames > 90) {
                box.grabbed = false
                box.pinchFrames = 0
                box.lostFrames = 0
              }
            }
          }

          // --- manual (touch/slider) override ---
          if (controlsNow.touchActive) {
            box.lightX = controlsNow.touchX
            box.lightY = controlsNow.touchY
            box.lightZ = controlsNow.lightZ
            box.everControlled = true
          }

          // --- idle orbit until first interaction (TypeGPU demo behavior) ---
          if (!box.everControlled && !box.grabbed) {
            const phase = performance.now() * 0.00024
            box.lightX = 0.5 + Math.cos(phase) * 0.26
            box.lightY = 0.44 + Math.sin(phase * 1.37) * 0.26 * 0.8
            box.lightZ = controlsNow.lightZ
          }
          box.lightZ = Math.min(Math.max(box.lightZ, LIGHT_Z_MIN), LIGHT_Z_MAX)

          // (The old z-floor sampled the CoreML depth buffer on the CPU;
          // with GPU-resident depth there is nothing to read - the direct
          // fingertip z control keeps the light in front of the hand, which
          // covers the physical case that mattered.)

          // --- bulb screen size ---
          // Hand-held: scale by the hand's angular size (Vision handSize is
          // proportional to 1/distance, the exact law the hand's own screen
          // size follows - so the bulb grows by the same factor the hand
          // does). Free-flying: perspective from a virtual camera close in
          // front of the scene (must stay above LIGHT_Z_MAX = 0.6).
          {
            const HAND_SIZE_REF = 0.17
            const BULB_CAMERA_Z = 0.85
            let sizeSource = 0
            if (controlsNow.handControl) {
              if (box.grabbed) {
                // Only the hand HOLDING the light may size it - an open
                // second hand moving toward the camera must not inflate it.
                sizeSource = box.grabbedHandSize
              } else {
                if (hand.hand1.tracked && hand.hand1.handSize > sizeSource) {
                  sizeSource = hand.hand1.handSize
                }
                if (hand.hand2.tracked && hand.hand2.handSize > sizeSource) {
                  sizeSource = hand.hand2.handSize
                }
              }
            }
            const targetScale = sizeSource > 0
              ? sizeSource / HAND_SIZE_REF
              : BULB_CAMERA_Z / Math.max(BULB_CAMERA_Z - box.lightZ, 0.2)
            const clamped = Math.min(Math.max(targetScale, 0.45), 3.5)
            box.bulbScale += (clamped - box.bulbScale) * 0.2
          }

          // --- relight uniforms ---
          const relightParams = new ArrayBuffer(96)
          const rpF32 = new Float32Array(relightParams)
          const rpU32 = new Uint32Array(relightParams)
          rpF32[0] = controlsNow.colorR
          rpF32[1] = controlsNow.colorG
          rpF32[2] = controlsNow.colorB
          rpF32[3] = 1
          rpF32[4] = box.lightX
          rpF32[5] = box.lightY
          rpF32[6] = box.lightZ
          rpF32[7] = controlsNow.exposure
          rpF32[8] = controlsNow.intensity
          rpF32[9] = controlsNow.relief
          rpF32[10] = controlsNow.specular
          rpF32[11] = controlsNow.shadow
          rpF32[12] = controlsNow.occlusion
          rpU32[13] = controlsNow.mode
          rpU32[14] = mirrored ? 1 : 0
          // World-space aspect: the displayed region is the 4:3 crop.
          rpF32[15] = DISPLAY_ASPECT
          rpF32[16] = cropW
          rpF32[17] = cropH
          rpF32[18] = cropOffX
          rpF32[19] = cropOffY
          rpF32[20] = (renderStart / 1000) % 1000
          rpF32[21] = box.bulbScale
          device.queue.writeBuffer(pipeline.relightParamsBuffer, 0, relightParams)

          const bindGroup = device.createBindGroup({
            layout: pipeline.relightPipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: pipeline.relightParamsBuffer } },
              { binding: 1, resource: pipeline.sampler },
              { binding: 2, resource: pipeline.surfaceView },
              { binding: 3, resource: externalTexture },
            ],
          })
          const pass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: pipeline.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
              },
            ],
          })
          pass.setPipeline(pipeline.relightPipeline)
          pass.setBindGroup(0, bindGroup)
          pass.draw(3)
          pass.end()
          const tSub0 = performance.now()
          device.queue.submit([encoder.finish()])
          pipeline.context.present()
          externalTexture.destroy()
          const tSub = performance.now() - tSub0
          box.tSync = box.tSync * 0.9 + tSync * 0.1
          box.tEnc = box.tEnc * 0.9 + tEnc * 0.1
          box.tSub = box.tSub * 0.9 + tSub * 0.1

          // --- stats ---
          box.frameCount += 1
          const now = performance.now()
          if (box.lastFrameTime > 0) {
            const dt = now - box.lastFrameTime
            box.fps = box.fps * 0.9 + (1000 / Math.max(dt, 1)) * 0.1
          }
          box.lastFrameTime = now
          if (box.frameCount % 150 === 0) {
            console.log(
              `[LightDemo] #${box.frameCount} ${box.fps.toFixed(0)}fps ` +
                `render=${(now - renderStart).toFixed(1)}ms ` +
                `sync=${box.tSync.toFixed(1)} enc=${box.tEnc.toFixed(1)} wait=${box.tWait.toFixed(1)} sub=${box.tSub.toFixed(1)} ` +
                `depth#${depth.seq}=${depth.inferenceTimeMs.toFixed(0)}ms ` +
                `hand#${hand.seq}=${hand.detectionTimeMs.toFixed(0)}ms ` +
                `hands=${(hand.hand1.tracked ? 1 : 0) + (hand.hand2.tracked ? 1 : 0)} ` +
                `pinch=${hand.hand1.pinchRatio.toFixed(2)} ` +
                `light=(${box.lightX.toFixed(2)},${box.lightY.toFixed(2)},${box.lightZ.toFixed(2)}) ` +
                `grabbed=${box.grabbed} rot=${rotationDeg} ` +
                `handSize=${hand.hand1.handSize.toFixed(3)} ` +
                `h1=(x${hand.hand1.midX.toFixed(2)} c${hand.hand1.confidence.toFixed(2)} ` +
                `d${hand.hand1.disparity.toFixed(3)} p${hand.hand1.pinchRatio.toFixed(2)}) ` +
                `h2=(x${hand.hand2.midX.toFixed(2)} c${hand.hand2.confidence.toFixed(2)} ` +
                `d${hand.hand2.disparity.toFixed(3)} p${hand.hand2.pinchRatio.toFixed(2)})`,
            )
          }
          if (box.frameCount % 15 === 0) {
            nitro.setStatus({
              frameCount: box.frameCount,
              fps: box.fps,
              renderTimeMs: now - renderStart,
              depthTimeMs: depth.inferenceTimeMs,
              handTimeMs: hand.detectionTimeMs,
              frameWidth: frame.width,
              frameHeight: frame.height,
              frameOrientation: frame.orientation,
              frameMirrored: frame.isMirrored,
              pixelFormat: frame.pixelFormat,
              lightX: box.lightX,
              lightY: box.lightY,
              lightZ: box.lightZ,
              handTracked: hand.hand1.tracked || hand.hand2.tracked,
              pinchRatio: Math.min(hand.hand1.pinchRatio, hand.hand2.pinchRatio),
              grabbed: box.grabbed,
              depthSeq: depth.seq,
              handSeq: hand.seq,
            })
          }
        } finally {
          videoFrame.release()
        }
      } catch (e) {
        console.log(`[LightDemo] frame error: ${String(e)}`)
      } finally {
        nativeBuffer.release()
        frame.dispose()
      }
    },
    [pipeline, device, nitro, depthart, gpuDepth, box, rnwgpu],
  )

  // Synchronous depth makes the frame callback take ~1 model interval, so
  // at 60fps camera every other frame is dropped by design - only log a
  // periodic summary.
  const dropBox = useMemo(() => ({ count: 0 }), [])
  const onFrameDropped = useCallback(
    (reason: string) => {
      'worklet'
      dropBox.count += 1
      if (dropBox.count % 300 === 0) {
        console.log(`[LightDemo] ${dropBox.count} frames dropped so far (${reason})`)
      }
    },
    [dropBox],
  )

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    targetResolution: TARGET_RESOLUTION,
    onFrame: onFrame,
    onFrameDropped: onFrameDropped,
  })

  useCamera({
    isActive: pipeline != null && nitro != null && cameraDevice != null && !cameraSuspended,
    device: cameraDevice as NonNullable<typeof cameraDevice>,
    outputs: [frameOutput],
    constraints: CAMERA_CONSTRAINTS,
    // Pin the output orientation to the sensor's native orientation so
    // Frames always arrive tagged 'up' - camera, depth and hand coordinates
    // then all share the same (landscape) space with no rotation anywhere.
    orientationSource: 'custom',
    onStarted: () => console.log('[LightDemo] camera started'),
    onStopped: () => console.log('[LightDemo] camera stopped'),
    onError: (e) => console.log(`[LightDemo] camera error: ${String(e)}`),
    onInterruptionStarted: (reason) =>
      console.log(`[LightDemo] camera interrupted: ${String(reason)}`),
    onInterruptionEnded: () => console.log('[LightDemo] camera interruption over'),
    onConfigured: () => console.log('[LightDemo] camera configured'),
  })
  useEffect(() => {
    // 'right' maps to AVCaptureVideoOrientation.landscapeLeft. This must be
    // a LANDSCAPE orientation: VisionCamera maps 'up' to .portrait, and
    // orientation-capable external cameras (the Insta360 gimbal) physically
    // rotate their sensor to match the connection's videoOrientation - a
    // portrait connection cost us 90deg-rotated buffers AND a narrow
    // portrait crop of the scene (compare FaceTime's wide landscape view).
    frameOutput.outputOrientation = 'right'
  }, [frameOutput])

  // Square-ish canvas matching the depth model aspect (4:3), centered.
  const aspect = nitro != null ? nitro.depthWidth / nitro.depthHeight : 4 / 3
  const maxW = window.width
  const maxH = window.height - 140
  const canvasW = Math.min(maxW, maxH * aspect)
  const canvasH = canvasW / aspect

  if (error != null) {
    return (
      <View style={styles.center}>
        <Text style={styles.error}>{error}</Text>
      </View>
    )
  }
  return (
    <View style={styles.root}>
      <View
        style={{ width: canvasW, height: canvasH }}
        onTouchStart={(e) => {
          const x = e.nativeEvent.locationX / canvasW
          const y = e.nativeEvent.locationY / canvasH
          setControls((c) => ({ ...c, touchX: x, touchY: y, touchActive: true }))
        }}
        onTouchMove={(e) => {
          const x = e.nativeEvent.locationX / canvasW
          const y = e.nativeEvent.locationY / canvasH
          setControls((c) => ({ ...c, touchX: x, touchY: y, touchActive: true }))
        }}
        onTouchEnd={() => {
          setControls((c) => ({ ...c, touchActive: false }))
        }}>
        <Canvas
          ref={ref}
          style={[
            styles.canvas,
            // Largest 4:3 rect that fits the window (minus HUD strip):
            // RN's aspectRatio+maxHeight combo clamps height AFTER
            // computing it, silently stretching the canvas.
            window.width / (window.height - 120) > 4 / 3
              ? { height: window.height - 120, width: ((window.height - 120) * 4) / 3 }
              : { width: window.width, height: (window.width * 3) / 4 },
          ]}
        />
        {(nitro == null || pipeline == null || (status?.frameCount ?? 0) === 0) && (
          <View style={styles.loadingOverlay} pointerEvents="none">
            <ActivityIndicator size="large" color="#ffffff" />
            <Text style={styles.loadingText}>
              {nitro == null
                ? 'Loading depth model…'
                : pipeline == null
                  ? 'Preparing GPU pipelines…'
                  : 'Starting camera…'}
            </Text>
          </View>
        )}
      </View>

      <View style={styles.toolbar}>
        <Pressable
          style={styles.button}
          onPress={() => setControls((c) => ({ ...c, mode: (c.mode + 1) % 4 }))}>
          <Text style={styles.buttonText}>{MODE_NAMES[controls.mode]}</Text>
        </Pressable>
        <Pressable
          style={styles.button}
          onPress={() => setControls((c) => ({ ...c, handControl: !c.handControl }))}>
          <Text style={styles.buttonText}>
            Hand: {controls.handControl ? 'on' : 'off'}
          </Text>
        </Pressable>
        <Pressable
          style={styles.button}
          onPress={() =>
            setControls((c) => {
              const steps = [-1, 0, 90, 180, 270]
              const next = steps[(steps.indexOf(c.rotationOverride) + 1) % steps.length]
              return { ...c, rotationOverride: next }
            })
          }>
          <Text style={styles.buttonText}>
            Rot: {controls.rotationOverride < 0 ? 'auto' : `${controls.rotationOverride}°`}
          </Text>
        </Pressable>
        <Pressable
          style={styles.button}
          onPress={() =>
            setControls((c) => ({ ...c, lightZ: Math.min(c.lightZ + 0.15, LIGHT_Z_MAX) }))
          }>
          <Text style={styles.buttonText}>Z+</Text>
        </Pressable>
        <Pressable
          style={styles.button}
          onPress={() =>
            setControls((c) => ({ ...c, lightZ: Math.max(c.lightZ - 0.15, LIGHT_Z_MIN) }))
          }>
          <Text style={styles.buttonText}>Z-</Text>
        </Pressable>
      </View>

      <Text style={styles.hud}>
        {status == null
          ? device == null
            ? 'Waiting for GPU device...'
            : nitro == null
              ? 'Loading CoreML model...'
              : pipeline == null
                ? 'Creating pipelines...'
                : cameraDevice == null
                  ? 'No camera device!'
                  : 'Waiting for frames...'
          : `${status.fps.toFixed(0)}fps render ${status.renderTimeMs.toFixed(1)}ms | ` +
            `depth ${status.depthTimeMs.toFixed(0)}ms #${status.depthSeq} | ` +
            `hand ${status.handTimeMs.toFixed(0)}ms ` +
            `${status.handTracked ? `pinch ${status.pinchRatio.toFixed(2)}` : 'none'}` +
            `${status.grabbed ? ' GRABBED' : ''} | ` +
            `${status.frameWidth}x${status.frameHeight} ${status.pixelFormat} ` +
            `${status.frameOrientation}${status.frameMirrored ? ' mirrored' : ''} | ` +
            `light (${status.lightX.toFixed(2)}, ${status.lightY.toFixed(2)}, ` +
            `${status.lightZ.toFixed(2)})`}
      </Text>
    </View>
  )
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: 'black',
    alignItems: 'center',
    justifyContent: 'center',
  },
  canvas: { alignSelf: 'center' },
  center: {
    flex: 1,
    backgroundColor: 'black',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  info: { color: 'white', fontSize: 16, marginBottom: 16 },
  error: { color: '#ff6666', fontSize: 14 },
  toolbar: { flexDirection: 'row', gap: 12, marginTop: 12 },
  button: {
    backgroundColor: '#222',
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 8,
  },
  buttonText: { color: 'white', fontSize: 14, fontWeight: '600' },
  hud: {
    color: '#8f8',
    fontSize: 11,
    fontFamily: 'Menlo',
    marginTop: 8,
    paddingHorizontal: 12,
    textAlign: 'center',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 14,
    backgroundColor: 'rgba(0, 0, 0, 0.55)',
  },
  loadingText: { color: 'white', fontSize: 15, fontWeight: '600' },
})
