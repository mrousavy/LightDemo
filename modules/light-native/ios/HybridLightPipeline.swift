//
//  HybridLightPipeline.swift
//  LightDemo
//

import Accelerate
import CoreImage
import CoreML
import CoreVideo
import Foundation
import NitroModules
import Vision
import VisionCamera

/// Frame analysis pipeline: CoreML monocular depth (ANE) + Vision hand pose.
/// Everything runs SYNCHRONOUSLY on the caller's (frame-processor) thread so
/// results always correspond to the exact frame being rendered.
final class HybridLightPipeline: HybridLightPipelineSpec {
  private let mlModel: MLModel
  private let modelInputName: String
  private let modelWidth: Int
  private let modelHeight: Int

  // Direct-CoreML preprocessing: GPU center-crop + scale into a reusable
  // BGRA buffer (Vision's VNCoreMLRequest costs ~23ms extra per frame on
  // the same model - measured 40ms vs 16.6ms direct on this M1 Max).
  // No color management (the model wants raw pixel values, and the working-
  // space conversion is pure overhead) + intermediate caching for repeated
  // identical graphs.
  private let ciContext = CIContext(options: [
    .workingColorSpace: NSNull(),
    .cacheIntermediates: true,
  ])
  private var inputBuffer: CVPixelBuffer?
  // Reused every prediction: MLFeatureValue holds the pixel buffer by
  // reference, so CoreML sees the freshly rendered contents each frame
  // without per-frame provider/NSObject allocations.
  private var inputProvider: MLDictionaryFeatureProvider?

  // Persistent Vision objects: VNSequenceRequestHandler caches state across
  // video frames and avoids the per-frame handler setup cost.
  private let handRequest: VNDetectHumanHandPoseRequest = {
    let request = VNDetectHumanHandPoseRequest()
    request.maximumHandCount = 2
    // Pin Vision's hand-pose networks OFF the Neural Engine: they otherwise
    // schedule onto it and CONTEND with our depth prediction running
    // concurrently (hands wall time ballooned 17.6ms -> 33ms overlapped).
    if #available(iOS 17.0, *) {
      let cpu = MLComputeDevice.allComputeDevices.first(where: {
        if case .cpu = $0 { return true }
        return false
      })
      if let cpu {
        request.setComputeDevice(cpu, for: .main)
        request.setComputeDevice(cpu, for: .postProcessing)
      }
    }
    return request
  }()
  private let sequenceHandler = VNSequenceRequestHandler()
  /// Runs Vision hand detection concurrently with the ANE depth prediction
  /// inside analyzeSync (joined before it returns).
  private let handQueue = DispatchQueue(label: "com.mrousavy.lightdemo.hands", qos: .userInteractive)

  // One persistent IOSurface-backed output buffer, handed to CoreML via
  // MLPredictionOptions.outputBackings - the model writes depth into the
  // SAME surface every frame, so the consumer can import it into WebGPU
  // once and reuse the texture. `latestOutput` normally IS `outputBuffer`;
  // it only differs if CoreML ever rejects the backing.
  private let outputBuffer: CVPixelBuffer
  private let modelOutputName: String
  private let predictionOptions: MLPredictionOptions

  private let lock = NSLock()
  private var latestOutput: CVPixelBuffer?
  private var depthSeq: Int = -1
  private var depthLow: Float = 0
  private var depthHigh: Float = 1
  private var depthTimeMs: Double = 0
  private var depthPrepMs: Double = 0
  private var depthPredictMs: Double = 0

  private var handSeq: Int = -1
  private var latestHand: HandResult
  private var handProbes: ([CGPoint], [CGPoint]) = ([], [])

  private var controls: LightControls
  private var status: LightStatus

  var depthWidth: Double { Double(modelWidth) }
  var depthHeight: Double { Double(modelHeight) }
  init(model: MLModel, inputWidth: Int, inputHeight: Int) throws {
    self.mlModel = model
    guard let inputName = model.modelDescription.inputDescriptionsByName.first(where: {
      $0.value.imageConstraint != nil
    })?.key else {
      throw RuntimeError.error(withMessage: "Depth model has no image input")
    }
    self.modelInputName = inputName
    self.modelWidth = inputWidth
    self.modelHeight = inputHeight

    guard let outputName = model.modelDescription.outputDescriptionsByName.first(where: {
      $0.value.type == .image
    })?.key else {
      throw RuntimeError.error(withMessage: "Depth model has no image output")
    }
    self.modelOutputName = outputName
    let outputAttributes: [CFString: Any] = [
      kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
      kCVPixelBufferMetalCompatibilityKey: true,
    ]
    var output: CVPixelBuffer?
    CVPixelBufferCreate(
      kCFAllocatorDefault, inputWidth, inputHeight,
      kCVPixelFormatType_OneComponent16Half,
      outputAttributes as CFDictionary, &output)
    guard let output else {
      throw RuntimeError.error(withMessage: "Failed to allocate depth output buffer")
    }
    self.outputBuffer = output
    let options = MLPredictionOptions()
    if #available(iOS 16.0, *) {
      options.outputBackings = [outputName: output]
    }
    self.predictionOptions = options

    self.latestHand = HandResult(
      seq: -1, hand1: Self.emptyHand, hand2: Self.emptyHand, detectionTimeMs: 0)
    self.controls = LightControls(
      mode: 0, intensity: 3.0, exposure: 0.5, relief: 0.85, specular: 0.22,
      shadow: 0.7, occlusion: 0.55, colorR: 1.0, colorG: 0.83, colorB: 0.6,
      touchX: 0.34, touchY: 0.34, touchActive: false, lightZ: 0.25,
      handControl: true, mirror: true, rotationOverride: -1, snapshotPath: "")
    self.status = LightStatus(
      frameCount: 0, fps: 0, renderTimeMs: 0, depthTimeMs: 0, handTimeMs: 0,
      frameWidth: 0, frameHeight: 0, frameOrientation: "", frameMirrored: false,
      pixelFormat: "", lightX: 0.34, lightY: 0.34, lightZ: 0.25,
      handTracked: false, pinchRatio: 1, grabbed: false, depthSeq: -1, handSeq: -1)
  }

  var memorySize: Int {
    // input BGRA + output f16 pixel buffers
    return modelWidth * modelHeight * (4 + 2)
  }

  func analyzeSync(
    frame: (any HybridFrameSpec), orientationDegrees: Double, runHands: Bool, runDepth: Bool
  ) throws -> DepthResult {
    // Typed Frame handoff: cast the spec to VisionCamera's public
    // NativeFrame protocol for native buffer access.
    guard let nativeFrame = frame as? any NativeFrame else {
      throw RuntimeError.error(withMessage: "The given Frame is not a NativeFrame!")
    }
    guard let sampleBuffer = nativeFrame.sampleBuffer,
          let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      throw RuntimeError.error(withMessage: "The given Frame has no valid pixel buffer!")
    }
    let orientation = Self.cgOrientation(fromDegrees: Int(orientationDegrees))

    // Prepare the shared model input once (upright + center-cropped), then
    // run the ANE depth prediction and Vision hand detection CONCURRENTLY:
    // depth occupies the Neural Engine while hands run on CPU, and both
    // only READ the prepared buffer. The join before returning keeps the
    // API fully synchronous - the frame just stops paying for the two
    // serially (at 448x336 that is 29.7ms + 17.6ms -> ~31ms).
    let start = CACurrentMediaTime()
    guard let input = prepareModelInput(from: pixelBuffer, orientation: orientation) else {
      print("[LightPipeline] depth: failed to prepare model input")
      return try getDepthResult()
    }
    let prepDone = CACurrentMediaTime()
    if runHands {
      let group = DispatchGroup()
      group.enter()
      handQueue.async { [self] in
        detectHands(onPreparedInput: input)
        group.leave()
      }
      if runDepth { predictDepth(on: input, start: start, prepDone: prepDone) }
      group.wait()
      if runDepth { attachHandDepth() }
    } else if runDepth {
      predictDepth(on: input, start: start, prepDone: prepDone)
    }

    return try getDepthResult()
  }

  /// Maps "degrees of rotation needed to display upright" to the EXIF-style
  /// orientation tag (same convention as VisionCamera's CameraOrientation).
  private static func cgOrientation(fromDegrees degrees: Int) -> CGImagePropertyOrientation {
    switch ((degrees % 360) + 360) % 360 {
    case 45..<135: return .right
    case 135..<225: return .down
    case 225..<315: return .left
    default: return .up
    }
  }

  // MARK: - Depth

  /// Upright `source` by `orientation`, center-crop to the model aspect, and
  /// scale into the reusable 392x294 BGRA input buffer on the GPU.
  private func prepareModelInput(
    from source: CVPixelBuffer, orientation: CGImagePropertyOrientation
  ) -> CVPixelBuffer? {
    if inputBuffer == nil {
      let attributes: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
        kCVPixelBufferMetalCompatibilityKey: true,
      ]
      CVPixelBufferCreate(
        kCFAllocatorDefault, modelWidth, modelHeight, kCVPixelFormatType_32BGRA,
        attributes as CFDictionary, &inputBuffer)
    }
    guard let target = inputBuffer else { return nil }

    var image = CIImage(cvPixelBuffer: source)
    if orientation != .up {
      image = image.oriented(orientation)
    }
    let extent = image.extent
    let modelAspect = CGFloat(modelWidth) / CGFloat(modelHeight)
    let sourceAspect = extent.width / extent.height
    var cropRect = extent
    if sourceAspect > modelAspect {
      cropRect.size.width = extent.height * modelAspect
      cropRect.origin.x = extent.origin.x + (extent.width - cropRect.size.width) / 2
    } else {
      cropRect.size.height = extent.width / modelAspect
      cropRect.origin.y = extent.origin.y + (extent.height - cropRect.size.height) / 2
    }
    let scale = CGFloat(modelWidth) / cropRect.width
    image = image
      .cropped(to: cropRect)
      .transformed(by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y))
      .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    ciContext.render(image, to: target)
    return target
  }

  private func predictDepth(
    on input: CVPixelBuffer, start: CFTimeInterval, prepDone: CFTimeInterval
  ) {
    let output: CVPixelBuffer
    do {
      if inputProvider == nil {
        inputProvider = try MLDictionaryFeatureProvider(
          dictionary: [modelInputName: MLFeatureValue(pixelBuffer: input)])
      }
      let prediction = try mlModel.prediction(from: inputProvider!, options: predictionOptions)
      guard let buffer = prediction.featureValue(for: modelOutputName)?.imageBufferValue
      else {
        print("[LightPipeline] depth: prediction has no image output")
        return
      }
      // Normally `buffer` IS our outputBacking; CoreML falls back to its own
      // pool only if the backing is unusable.
      output = buffer
    } catch {
      print("[LightPipeline] depth inference failed: \(error)")
      return
    }
    let predictDone = CACurrentMediaTime()
    let outWidth = CVPixelBufferGetWidth(output)
    let outHeight = CVPixelBufferGetHeight(output)
    guard outWidth == modelWidth, outHeight == modelHeight else {
      print("[LightPipeline] depth: unexpected output size \(outWidth)x\(outHeight)")
      return
    }
    guard CVPixelBufferGetIOSurface(output) != nil else {
      print("[LightPipeline] depth: output is not IOSurface-backed")
      return
    }

    // Range from a sparse 32x32 grid probe (two scalars out; the full-frame
    // pixel path stays on the GPU - JS imports the IOSurface directly).
    let (low, high) = Self.sparseRobustRange(of: output)
    let elapsed = (CACurrentMediaTime() - start) * 1000

    lock.lock()
    latestOutput = output
    depthSeq += 1
    depthLow = low
    depthHigh = high
    depthTimeMs = elapsed
    depthPrepMs = (prepDone - start) * 1000
    depthPredictMs = (predictDone - prepDone) * 1000
    lock.unlock()
  }

  /// 2nd/98th percentile of the f16 depth map, estimated from a sparse 32x32
  /// sample grid (1024 texels) - statistically equivalent for the smooth
  /// depth fields this model produces, without a full-frame CPU read.
  private static func sparseRobustRange(of buffer: CVPixelBuffer) -> (Float, Float) {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return (0, 1) }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

    let grid = 32
    var samples = [Float]()
    samples.reserveCapacity(grid * grid)
    for gy in 0..<grid {
      let y = (gy * height) / grid + height / (grid * 2)
      let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float16.self)
      for gx in 0..<grid {
        let x = (gx * width) / grid + width / (grid * 2)
        let value = Float(row[x])
        if value.isFinite {
          samples.append(value)
        }
      }
    }
    guard samples.count > 8 else { return (0, 1) }
    samples.sort()
    let low = samples[max(Int(Double(samples.count) * 0.02), 0)]
    var high = samples[min(Int(Double(samples.count) * 0.98), samples.count - 1)]
    high = max(high, low + 0.001)
    return (low, high)
  }

  func sampleDepthMax(points: [Double]) throws -> Double {
    lock.lock()
    let output = latestOutput
    lock.unlock()
    guard let buffer = output else { return -1 }
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return -1 }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

    var nearest = -Float.greatestFiniteMagnitude
    var found = false
    let offsets = [(0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)]
    var i = 0
    while i + 1 < points.count {
      let cx = min(max(Int(points[i] * Double(width)), 2), width - 3)
      let cy = min(max(Int(points[i + 1] * Double(height)), 2), height - 3)
      for (ox, oy) in offsets {
        let row = base.advanced(by: (cy + oy) * bytesPerRow)
          .assumingMemoryBound(to: Float16.self)
        let value = Float(row[cx + ox])
        if value.isFinite, value > nearest {
          nearest = value
          found = true
        }
      }
      i += 2
    }
    return found ? Double(nearest) : -1
  }

  func getDepthResult() throws -> DepthResult {
    lock.lock()
    let seq = depthSeq
    let low = depthLow
    let high = depthHigh
    let time = depthTimeMs
    let prep = depthPrepMs
    let predict = depthPredictMs
    let output = latestOutput
    lock.unlock()
    // The IOSurface pointer stays valid as long as `latestOutput` is
    // retained, i.e. until the next inference replaces it - consumers must
    // import it within the same frame.
    var surfacePointer: UInt64 = 0
    if let output, let surface = CVPixelBufferGetIOSurface(output) {
      surfacePointer = UInt64(UInt(bitPattern: Unmanaged.passUnretained(
        surface.takeUnretainedValue()).toOpaque()))
    }
    return DepthResult(
      seq: Double(seq), width: Double(modelWidth), height: Double(modelHeight),
      low: Double(low), high: Double(high), inferenceTimeMs: time,
      prepTimeMs: prep, predictTimeMs: predict, surfacePointer: surfacePointer)
  }

  // MARK: - Hands

  private static let emptyHand = TrackedHand(
    tracked: false, thumbX: 0, thumbY: 0, indexX: 0, indexY: 0,
    midX: 0, midY: 0, pinchRatio: 1, handSize: 0, confidence: 0, disparity: -1)

  /// Extract a TrackedHand from a Vision observation. The input already IS
  /// the upright center crop, so only the y-flip is needed (Vision uses a
  /// bottom-left origin).
  private static func extractHand(
    _ hand: VNHumanHandPoseObservation
  ) -> (hand: TrackedHand, probes: [CGPoint])? {
    guard let thumb = try? hand.recognizedPoint(.thumbTip),
          let index = try? hand.recognizedPoint(.indexTip),
          let wrist = try? hand.recognizedPoint(.wrist),
          let middleMCP = try? hand.recognizedPoint(.middleMCP),
          thumb.confidence > 0.3, index.confidence > 0.3
    else { return nil }
    let thumbX = Double(thumb.location.x)
    let thumbY = Double(1 - thumb.location.y)
    let indexX = Double(index.location.x)
    let indexY = Double(1 - index.location.y)
    let handSize = max(
      hypot(wrist.location.x - middleMCP.location.x,
            wrist.location.y - middleMCP.location.y),
      1e-4)
    let pinchDistance = hypot(thumb.location.x - index.location.x,
                              thumb.location.y - index.location.y)
    // Every confident landmark (y-flipped into crop space) for the robust
    // hand-depth probe in attachHandDepth().
    var probes: [CGPoint] = []
    if let all = try? hand.recognizedPoints(.all) {
      for (_, point) in all where point.confidence > 0.3 {
        probes.append(CGPoint(x: point.location.x, y: 1 - point.location.y))
      }
    }
    let tracked = TrackedHand(
      tracked: true, thumbX: thumbX, thumbY: thumbY,
      indexX: indexX, indexY: indexY,
      midX: (thumbX + indexX) / 2, midY: (thumbY + indexY) / 2,
      pinchRatio: Double(pinchDistance / handSize),
      handSize: Double(handSize),
      confidence: Double(min(thumb.confidence, index.confidence)),
      disparity: -1)
    return (tracked, probes)
  }

  /// Synchronous hand-pose detection (up to two hands) on the prepared
  /// (upright, cropped) model input buffer - landmarks come back directly
  /// in crop space. Slot order is not stable across frames.
  private func detectHands(onPreparedInput input: CVPixelBuffer) {
    let start = CACurrentMediaTime()
    do {
      try sequenceHandler.perform([handRequest], on: input)
    } catch {
      // fall through with no results
    }
    let extracted = (handRequest.results ?? []).compactMap(Self.extractHand)
    let elapsed = (CACurrentMediaTime() - start) * 1000
    lock.lock()
    handSeq += 1
    latestHand = HandResult(
      seq: Double(handSeq),
      hand1: extracted.count > 0 ? extracted[0].hand : Self.emptyHand,
      hand2: extracted.count > 1 ? extracted[1].hand : Self.emptyHand,
      detectionTimeMs: elapsed)
    handProbes = (
      extracted.count > 0 ? extracted[0].probes : [],
      extracted.count > 1 ? extracted[1].probes : [])
    lock.unlock()
  }

  /// Robust hand depth: raw disparity at the 85th percentile over every
  /// confident landmark. Runs AFTER the depth prediction and hand detection
  /// join (both buffers are stable), patching the disparity into the stored
  /// hand result.
  private func attachHandDepth() {
    lock.lock()
    let (probes1, probes2) = handProbes
    let output = latestOutput
    let hand = latestHand
    lock.unlock()
    guard let buffer = output, hand.hand1.tracked else { return }
    let d1 = Self.robustDisparity(of: probes1, in: buffer)
    let d2 = hand.hand2.tracked ? Self.robustDisparity(of: probes2, in: buffer) : -1
    let h1 = Self.withDisparity(hand.hand1, d1)
    let h2 = Self.withDisparity(hand.hand2, d2)
    lock.lock()
    latestHand = HandResult(
      seq: hand.seq, hand1: h1, hand2: h2, detectionTimeMs: hand.detectionTimeMs)
    lock.unlock()
  }

  private static func withDisparity(_ hand: TrackedHand, _ disparity: Double) -> TrackedHand {
    return TrackedHand(
      tracked: hand.tracked, thumbX: hand.thumbX, thumbY: hand.thumbY,
      indexX: hand.indexX, indexY: hand.indexY,
      midX: hand.midX, midY: hand.midY,
      pinchRatio: hand.pinchRatio, handSize: hand.handSize,
      confidence: hand.confidence, disparity: disparity)
  }

  private static func robustDisparity(of probes: [CGPoint], in buffer: CVPixelBuffer) -> Double {
    guard !probes.isEmpty else { return -1 }
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return -1 }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
    var values: [Float] = []
    values.reserveCapacity(probes.count)
    for probe in probes {
      let x = min(max(Int(probe.x * CGFloat(width)), 0), width - 1)
      let y = min(max(Int(probe.y * CGFloat(height)), 0), height - 1)
      let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float16.self)
      let value = Float(row[x])
      if value.isFinite { values.append(value) }
    }
    guard !values.isEmpty else { return -1 }
    // MAX, not a percentile: against depth-cue-heavy backgrounds (painted
    // murals) the model can sink MOST of a hand into the wall - if any
    // landmark still reads near, that is the hand's true plane. Spurious
    // too-near outliers are rare (model errors bleed toward background).
    return Double(values.max()!)
  }

  func getHandResult() throws -> HandResult {
    lock.lock()
    let hand = latestHand
    lock.unlock()
    return hand
  }

  // MARK: - Cross-runtime parameter/status store

  func setControls(controls: LightControls) throws {
    lock.lock()
    self.controls = controls
    lock.unlock()
  }

  func getControls() throws -> LightControls {
    lock.lock()
    let value = controls
    lock.unlock()
    return value
  }

  func setStatus(status: LightStatus) throws {
    lock.lock()
    self.status = status
    lock.unlock()
  }

  func getStatus() throws -> LightStatus {
    lock.lock()
    let value = status
    lock.unlock()
    return value
  }
}
