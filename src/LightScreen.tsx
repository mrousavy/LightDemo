import React, { useEffect, useMemo, useState } from 'react'
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
  mode: 0,
  intensity: 3.0,
  exposure: 0.5,
  relief: 0.85,
  specular: 0.22,
  shadow: 0.7,
  occlusion: 0.55,
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

  // 5. Poll status for the HUD
  useEffect(() => {
    if (nitro == null) return
    const interval = setInterval(() => setStatus(nitro.getStatus()), 500)
    return () => clearInterval(interval)
  }, [nitro])

  const devices = useCameraDevices()
  const cameraDevice = useMemo(
    () =>
      devices.find((d) => d.position === 'front') ??
      devices.find((d) => d.position === 'back') ??
      devices[0],
    [devices],
  )

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
    }),
    [],
  )

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    onFrame: (frame) => {
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
      const nativeBuffer = frame.getNativeBuffer()
      try {
        // Kick depth + hand analysis (async, drop-if-busy).
        nitro.submitFrame(nativeBuffer.pointer, true, controlsNow.handControl)

        const videoFrame = rnwgpu.createVideoFrameFromNativeBuffer(nativeBuffer.pointer)
        try {
          const mirrored = frame.isMirrored
          let rotationDeg: 0 | 90 | 180 | 270 = 0
          if (frame.orientation === 'right') rotationDeg = 90
          else if (frame.orientation === 'down') rotationDeg = 180
          else if (frame.orientation === 'left') rotationDeg = 270
          const rotated = rotationDeg === 90 || rotationDeg === 270
          const dispW = rotated ? videoFrame.height : videoFrame.width
          const dispH = rotated ? videoFrame.width : videoFrame.height

          const encoder = device.createCommandEncoder()

          // --- depth passes (only when a new inference completed) ---
          const depth = nitro.getDepthResult()
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
          const hand = nitro.getHandResult()
          if (controlsNow.handControl && hand.seq >= 0 && hand.seq !== box.lastHandSeq) {
            box.lastHandSeq = hand.seq
            if (hand.tracked) {
              // Hand coords are in crop space (buffer orientation); mirror
              // into display space to compare with the light.
              const handX = mirrored ? 1 - hand.midX : hand.midX
              const handY = hand.midY
              const isPinching = hand.pinchRatio < 0.28
              const isOpen = hand.pinchRatio > 0.42
              if (!box.grabbed) {
                const dx = handX - box.lightX
                const dy = handY - box.lightY
                const near = dx * dx + dy * dy < 0.22 * 0.22
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
                  // Light depth follows the hand's depth in the scene.
                  const px = Math.min(
                    Math.max(Math.floor(hand.midX * pipeline.depthW), 0),
                    pipeline.depthW - 1,
                  )
                  const py = Math.min(
                    Math.max(Math.floor(hand.midY * pipeline.depthH), 0),
                    pipeline.depthH - 1,
                  )
                  const disparity = new Float32Array(depth.data)
                  const d = disparity[py * pipeline.depthW + px]
                  if (d === d) {
                    const span = Math.max(box.rangeHigh - box.rangeLow, 0.001)
                    const normalized = Math.min(
                      Math.max((d - box.rangeLow) / span, 0),
                      1,
                    )
                    const targetZ = -0.55 + normalized * (0.9 + 0.55) + 0.06
                    box.lightZ += (targetZ - box.lightZ) * 0.35
                  }
                }
              }
            } else {
              box.pinchFrames = 0
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
  })

  useCamera({
    isActive: pipeline != null && nitro != null && cameraDevice != null,
    device: cameraDevice as NonNullable<typeof cameraDevice>,
    outputs: [frameOutput],
  })

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
