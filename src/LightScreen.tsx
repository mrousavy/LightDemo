import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
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
import { DEPTH_PREPARE_SHADER, RELIGHT_SHADER, SURFACE_SHADER } from './shaders'

const REQUIRED_FEATURES: GPUFeatureName[] = [
  'rnwebgpu/native-texture' as GPUFeatureName,
  'dawn-multi-planar-formats' as GPUFeatureName,
]

// Light z range (surface-Z space), from the TypeGPU demo.
const LIGHT_Z_MIN = -0.66
const LIGHT_Z_MAX = 1.65

const DEFAULT_CONTROLS: LightControls = {
  mirror: true,
  // 270 = visually verified for the Insta360 Link 2 Pro's current mounting.
  // Cycle the Rot button to "auto" for face-roll-based auto-calibration.
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
  lightZ: 0.42,
  handControl: true,
  snapshotPath: '',
}

const MODE_NAMES = ['Relit', 'Camera', 'Depth', 'Normals']

interface PipelineState {
  context: RNCanvasContext
  sampler: GPUSampler
  depthPreparePipeline: GPUComputePipeline
  surfacePipeline: GPUComputePipeline
  relightPipeline: GPURenderPipeline
  computeParamsBuffer: GPUBuffer
  relightParamsBuffer: GPUBuffer
  disparityBuffer: GPUBuffer
  depthPrepareBindGroup: GPUBindGroup
  surfaceBindGroup: GPUBindGroup
  surfaceView: GPUTextureView
  depthW: number
  depthH: number
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

  // 3. WebGPU pipelines (device + canvas + depth size ready)
  useEffect(() => {
    if (device == null || nitro == null || pipeline != null) return
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

    const depthW = nitro.depthWidth
    const depthH = nitro.depthHeight
    const texelCount = depthW * depthH

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
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const relightParamsBuffer = device.createBuffer({
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const disparityBuffer = device.createBuffer({
      size: texelCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const historyBuffer = device.createBuffer({
      size: texelCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    const surfaceTexture = device.createTexture({
      size: [depthW, depthH],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    const surfaceView = surfaceTexture.createView()

    const depthPrepareBindGroup = device.createBindGroup({
      layout: depthPreparePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: computeParamsBuffer } },
        { binding: 1, resource: { buffer: disparityBuffer } },
        { binding: 2, resource: { buffer: historyBuffer } },
      ],
    })
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
      disparityBuffer,
      depthPrepareBindGroup,
      surfaceBindGroup,
      surfaceView,
      depthW,
      depthH,
    })
  }, [device, nitro, pipeline, ref])

  // 4. Push UI controls into the native store (worklet reads them per frame)
  useEffect(() => {
    nitro?.setControls(controls)
  }, [nitro, controls])

  // Debug handle for the CDP console (scripts/jsconsole.mjs).
  useEffect(() => {
    ;(globalThis as Record<string, unknown>).__nitro = nitro
  }, [nitro])

  // 5. Poll status for the HUD
  useEffect(() => {
    if (nitro == null) return
    const interval = setInterval(() => setStatus(nitro.getStatus()), 500)
    return () => clearInterval(interval)
  }, [nitro])

  // 6. Debug: periodic window snapshot into Documents/snap.png so rendering
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
  // then the built-in front camera.
  const cameraDevice = useMemo(() => {
    const real = devices.filter((d) => !d.isContinuityCamera && d.type !== 'continuity')
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
      lightZ: 0.42,
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
    }),
    [],
  )

  // Stable worklet: identity must only change when the captured pipeline
  // objects change, otherwise every React render re-serializes the closure
  // and resets the worklet-side `box` state.
  const onFrame = useCallback(
    (frame: Frame) => {
      'worklet'
      if (pipeline == null || device == null || nitro == null) {
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
      let rotationDeg: 0 | 90 | 180 | 270 = 0
      const faceRoll = nitro.lastFaceRollDegrees
      if (controlsNow.rotationOverride >= 0) {
        rotationDeg = controlsNow.rotationOverride as 0 | 90 | 180 | 270
      } else if (faceRoll > -900) {
        // Face-calibrated orientation: the roll of a detected face in the
        // raw buffer tells us how the buffer is rotated (the VisionCamera
        // tag comes from the connection's default portrait videoOrientation
        // and is wrong for external/gimbal cameras). ROLL_SIGN converts
        // Vision's y-up roll into our display-rotation direction. Validated
        // live: roll=-90 measured while the verified rotation was 270.
        const ROLL_SIGN = 1
        const quantized =
          ((Math.round((ROLL_SIGN * faceRoll) / 90) * 90) % 360 + 360) % 360
        rotationDeg = quantized as 0 | 90 | 180 | 270
      } else if (frame.orientation === 'right') rotationDeg = 90
      else if (frame.orientation === 'down') rotationDeg = 180
      else if (frame.orientation === 'left') rotationDeg = 270
      const nativeBuffer = frame.getNativeBuffer()
      try {
        // Kick hand analysis (async, drop-if-busy). Depth runs synchronously
        // below so lighting always matches this exact frame.
        nitro.submitFrame(nativeBuffer.pointer, rotationDeg, false, controlsNow.handControl)

        const videoFrame = rnwgpu.createVideoFrameFromNativeBuffer(nativeBuffer.pointer)
        try {
          // XOR: mirror the display when exactly one of (buffer mirrored,
          // user wants mirror) holds. Applied consistently to the camera
          // fetch, the surface texture, and the hand coordinates.
          const mirrored = frame.isMirrored !== controlsNow.mirror
          const rotated = rotationDeg === 90 || rotationDeg === 270
          const dispW = rotated ? videoFrame.height : videoFrame.width
          const dispH = rotated ? videoFrame.width : videoFrame.height

          const encoder = device.createCommandEncoder()

          // --- depth: synchronous, same-frame (async depth lags the camera
          // image and paints ghost trails behind fast-moving objects) ---
          const depth = nitro.runDepthSync(nativeBuffer.pointer, rotationDeg)
          let reset = 0
          if (depth.seq >= 0 && depth.seq !== box.lastDepthSeq) {
            box.lastDepthSeq = depth.seq
            if (!box.rangeInitialized) {
              box.rangeInitialized = true
              box.rangeLow = depth.low
              box.rangeHigh = depth.high
              reset = 1
            } else {
              // RANGE_BLEND = 0.12 (TypeGPU demo)
              box.rangeLow = box.rangeLow + (depth.low - box.rangeLow) * 0.12
              box.rangeHigh = box.rangeHigh + (depth.high - box.rangeHigh) * 0.12
            }
            device.queue.writeBuffer(pipeline.disparityBuffer, 0, depth.data)
            const computeParams = new ArrayBuffer(32)
            const cpU32 = new Uint32Array(computeParams)
            const cpF32 = new Float32Array(computeParams)
            cpU32[0] = pipeline.depthW
            cpU32[1] = pipeline.depthH
            cpU32[2] = reset
            cpU32[3] = mirrored ? 1 : 0
            cpF32[4] = box.rangeLow
            cpF32[5] = box.rangeHigh
            device.queue.writeBuffer(pipeline.computeParamsBuffer, 0, computeParams)

            const compute = encoder.beginComputePass()
            compute.setPipeline(pipeline.depthPreparePipeline)
            compute.setBindGroup(0, pipeline.depthPrepareBindGroup)
            compute.dispatchWorkgroups(Math.ceil((pipeline.depthW * pipeline.depthH) / 64))
            compute.setPipeline(pipeline.surfacePipeline)
            compute.setBindGroup(0, pipeline.surfaceBindGroup)
            compute.dispatchWorkgroups(
              Math.ceil(pipeline.depthW / 8),
              Math.ceil(pipeline.depthH / 8),
            )
            compute.end()
          }

          // --- hand interaction ---
          const prevLightX = box.lightX
          const prevLightY = box.lightY
          let freshHandUpdate = false
          const hand = nitro.getHandResult()
          if (controlsNow.handControl && hand.seq >= 0 && hand.seq !== box.lastHandSeq) {
            box.lastHandSeq = hand.seq
            // Gate low-confidence detections, and reject single-frame
            // midpoint teleports (Vision jitters under motion blur, which
            // previously made the grabbed light jump across the screen).
            const rawX = mirrored ? 1 - hand.midX : hand.midX
            const rawY = hand.midY
            const jump = box.handValid
              ? Math.hypot(rawX - box.handX, rawY - box.handY)
              : 0
            const isOutlier = box.handValid && jump > 0.3 && box.outlierCount < 3
            const usable = hand.tracked && hand.confidence > 0.45 && !isOutlier
            if (hand.tracked && isOutlier) box.outlierCount += 1
            if (usable) {
              box.outlierCount = 0
              if (box.handValid) {
                box.handX += (rawX - box.handX) * 0.6
                box.handY += (rawY - box.handY) * 0.6
                box.pinchSmoothed += (hand.pinchRatio - box.pinchSmoothed) * 0.5
              } else {
                box.handX = rawX
                box.handY = rawY
                box.pinchSmoothed = hand.pinchRatio
                box.handValid = true
              }
              const handX = box.handX
              const handY = box.handY
              const isPinching = box.pinchSmoothed < 0.28
              const isOpen = box.pinchSmoothed > 0.42
              // Hover-steering (like the TypeGPU demo's pointer): a visible,
              // non-pinching hand gently attracts the light, making the
              // pinch-grab easy to initiate.
              if (!box.grabbed && !controlsNow.touchActive && !isPinching) {
                box.everControlled = true
                box.lightX += (handX - box.lightX) * 0.06
                box.lightY += (handY - box.lightY) * 0.06
                freshHandUpdate = true
              }
              if (!box.grabbed) {
                const dx = handX - box.lightX
                const dy = handY - box.lightY
                const near = dx * dx + dy * dy < 0.3 * 0.3
                if (isPinching && near) {
                  box.pinchFrames += 1
                  if (box.pinchFrames >= 2) {
                    box.grabbed = true
                    box.everControlled = true
                    box.releaseFrames = 0
                  }
                } else {
                  box.pinchFrames = 0
                }
              } else {
                if (isOpen) {
                  box.releaseFrames += 1
                  if (box.releaseFrames >= 2) {
                    box.grabbed = false
                    box.pinchFrames = 0
                  }
                } else {
                  box.releaseFrames = 0
                }
                if (box.grabbed) {
                  // Smoothly follow the pinch point.
                  box.lightX += (handX - box.lightX) * 0.5
                  box.lightY += (handY - box.lightY) * 0.5
                  freshHandUpdate = true
                  // Light depth = the FINGERTIPS' depth. The pinch midpoint
                  // alone often lands on the background peeking between the
                  // fingers, which pushed the light a few cm behind the hand
                  // and made it glitch - so sample small neighborhoods
                  // around thumb tip, index tip and midpoint and take the
                  // NEAREST disparity (the fingers are the nearest surface).
                  const disparity = new Float32Array(depth.data)
                  const w = pipeline.depthW
                  const h = pipeline.depthH
                  let nearest = Number.NEGATIVE_INFINITY
                  const points = [
                    hand.thumbX, hand.thumbY,
                    hand.indexX, hand.indexY,
                    hand.midX, hand.midY,
                  ]
                  for (let p = 0; p < 6; p += 2) {
                    const cx = Math.min(Math.max(Math.floor(points[p] * w), 2), w - 3)
                    const cy = Math.min(Math.max(Math.floor(points[p + 1] * h), 2), h - 3)
                    for (let t = 0; t < 5; t++) {
                      const ox = t === 1 ? -2 : t === 2 ? 2 : 0
                      const oy = t === 3 ? -2 : t === 4 ? 2 : 0
                      const d = disparity[(cy + oy) * w + (cx + ox)]
                      if (d === d && d > nearest) nearest = d
                    }
                  }
                  if (nearest > Number.NEGATIVE_INFINITY) {
                    const span = Math.max(box.rangeHigh - box.rangeLow, 0.001)
                    const normalized = Math.min(
                      Math.max((nearest - box.rangeLow) / span, 0),
                      1,
                    )
                    // Place the light at the fingertips' depth within the
                    // scene's own z range (surfaceZ space [-0.7, 0], small
                    // forward bias so the bulb sits between the fingers
                    // instead of embedded in them). Staying inside the scene
                    // range is what lets nearer surfaces (e.g. you stepping
                    // in front of the bulb) actually occlude it.
                    const targetZ = -0.68 + normalized * 0.7 + 0.03
                    box.lightZ += (targetZ - box.lightZ) * 0.25
                  }
                }
              }
            } else if (!hand.tracked) {
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

          // --- relight uniforms ---
          const modelAspect = pipeline.depthW / pipeline.depthH
          const frameAspect = dispW / dispH
          let cropW = 1
          let cropH = 1
          if (frameAspect > modelAspect) {
            cropW = modelAspect / frameAspect
          } else {
            cropH = frameAspect / modelAspect
          }
          const relightParams = new ArrayBuffer(80)
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
          rpF32[15] = 0
          rpF32[16] = cropW
          rpF32[17] = cropH
          rpF32[18] = (1 - cropW) / 2
          rpF32[19] = (1 - cropH) / 2
          device.queue.writeBuffer(pipeline.relightParamsBuffer, 0, relightParams)

          const externalTexture = device.importExternalTexture({
            source: videoFrame,
            label: 'camera-frame',
            rotation: rotationDeg,
          })
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
          device.queue.submit([encoder.finish()])
          pipeline.context.present()
          externalTexture.destroy()

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
                `depth#${depth.seq}=${depth.inferenceTimeMs.toFixed(0)}ms ` +
                `hand#${hand.seq}=${hand.detectionTimeMs.toFixed(0)}ms ` +
                `tracked=${hand.tracked} pinch=${hand.pinchRatio.toFixed(2)} ` +
                `light=(${box.lightX.toFixed(2)},${box.lightY.toFixed(2)},${box.lightZ.toFixed(2)}) ` +
                `grabbed=${box.grabbed} rot=${rotationDeg} roll=${faceRoll.toFixed(0)}`,
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
              handTracked: hand.tracked,
              pinchRatio: hand.pinchRatio,
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
    [pipeline, device, nitro, box, rnwgpu],
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
    onFrame: onFrame,
    onFrameDropped: onFrameDropped,
  })

  useCamera({
    isActive: pipeline != null && nitro != null && cameraDevice != null,
    device: cameraDevice as NonNullable<typeof cameraDevice>,
    outputs: [frameOutput],
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
    frameOutput.outputOrientation = 'up'
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
        <Canvas ref={ref} style={styles.canvas} />
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
  canvas: { flex: 1 },
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
})
