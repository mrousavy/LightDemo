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

/// Frame analysis pipeline: CoreML monocular depth (ANE) + Vision hand pose.
/// Both tasks run on their own serial queues with drop-if-busy semantics.
final class HybridLightPipeline: HybridLightPipelineSpec {
  private let mlModel: MLModel
  private let modelInputName: String
  private let modelWidth: Int
  private let modelHeight: Int

  // Direct-CoreML preprocessing: GPU center-crop + scale into a reusable
  // BGRA buffer (Vision's VNCoreMLRequest costs ~23ms extra per frame on
  // the same model - measured 40ms vs 16.6ms direct on this M1 Max).
  private let ciContext = CIContext(options: [.cacheIntermediates: false])
  private var inputBuffer: CVPixelBuffer?

  private let depthQueue = DispatchQueue(label: "light.depth", qos: .userInteractive)
  private let handQueue = DispatchQueue(label: "light.hands", qos: .userInteractive)
  private let depthBusy = NitroAtomicFlag()
  private let handBusy = NitroAtomicFlag()

  // Ping-pong Float32 depth buffers. `frontIndex` is the completed one.
  private let lock = NSLock()
  private var buffers: [UnsafeMutablePointer<Float>]
  private var frontIndex = 0
  private var depthSeq: Int = -1
  private var depthLow: Float = 0
  private var depthHigh: Float = 1
  private var depthTimeMs: Double = 0
  private var depthPrepMs: Double = 0
  private var depthPredictMs: Double = 0

  private var handSeq: Int = -1
  private var latestHand: HandResult

  // Face-based orientation auto-calibration (see spec docs).
  private var orientationScanCountdown = 0
  private var lastRollDeg: Double = -999

  private var controls: LightControls
  private var status: LightStatus

  var depthWidth: Double { Double(modelWidth) }
  var depthHeight: Double { Double(modelHeight) }
  var lastFaceRollDegrees: Double {
    lock.lock()
    defer { lock.unlock() }
    return lastRollDeg
  }

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
    let count = inputWidth * inputHeight
    self.buffers = [
      UnsafeMutablePointer<Float>.allocate(capacity: count),
      UnsafeMutablePointer<Float>.allocate(capacity: count),
    ]
    self.buffers[0].initialize(repeating: 0, count: count)
    self.buffers[1].initialize(repeating: 0, count: count)
    self.latestHand = HandResult(
      seq: -1, tracked: false, thumbX: 0, thumbY: 0, indexX: 0, indexY: 0,
      midX: 0, midY: 0, pinchRatio: 1, confidence: 0, detectionTimeMs: 0)
    self.controls = LightControls(
      mode: 0, intensity: 3.0, exposure: 0.5, relief: 0.85, specular: 0.22,
      shadow: 0.7, occlusion: 0.55, colorR: 1.0, colorG: 0.83, colorB: 0.6,
      touchX: 0.34, touchY: 0.34, touchActive: false, lightZ: 0.42,
      handControl: true, mirror: true, rotationOverride: -1, snapshotPath: "")
    self.status = LightStatus(
      frameCount: 0, fps: 0, renderTimeMs: 0, depthTimeMs: 0, handTimeMs: 0,
      frameWidth: 0, frameHeight: 0, frameOrientation: "", frameMirrored: false,
      pixelFormat: "", lightX: 0.34, lightY: 0.34, lightZ: 0.42,
      handTracked: false, pinchRatio: 1, grabbed: false, depthSeq: -1, handSeq: -1)
  }

  deinit {
    buffers[0].deallocate()
    buffers[1].deallocate()
  }

  var memorySize: Int {
    return modelWidth * modelHeight * 4 * 2
  }

  func submitFrame(
    pointer: UInt64, orientationDegrees: Double, runDepth: Bool, runHands: Bool
  ) throws {
    guard let raw = UnsafeRawPointer(bitPattern: UInt(pointer)) else {
      throw RuntimeError.error(withMessage: "submitFrame: pointer is null")
    }
    let orientation = Self.cgOrientation(fromDegrees: Int(orientationDegrees))

    if runDepth, !depthBusy.testAndSet() {
      // Retain the buffer for the async task; released when done.
      let retained = Unmanaged<CVPixelBuffer>.fromOpaque(raw).retain().takeUnretainedValue()
      depthQueue.async { [weak self] in
        defer {
          Unmanaged.passUnretained(retained).release()
          self?.depthBusy.clear()
        }
        self?.runDepthInference(on: retained, orientation: orientation)
      }
    }
    if runHands, !handBusy.testAndSet() {
      let retained = Unmanaged<CVPixelBuffer>.fromOpaque(raw).retain().takeUnretainedValue()
      handQueue.async { [weak self] in
        defer {
          Unmanaged.passUnretained(retained).release()
          self?.handBusy.clear()
        }
        self?.runHandDetection(on: retained, orientation: orientation)
      }
    }
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

  private func runDepthInference(
    on pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation
  ) {
    let start = CACurrentMediaTime()
    guard let input = prepareModelInput(from: pixelBuffer, orientation: orientation) else {
      print("[LightPipeline] depth: failed to prepare model input")
      return
    }
    let prepDone = CACurrentMediaTime()
    let output: CVPixelBuffer
    do {
      let provider = try MLDictionaryFeatureProvider(
        dictionary: [modelInputName: MLFeatureValue(pixelBuffer: input)])
      let prediction = try mlModel.prediction(from: provider)
      guard let buffer = prediction.featureNames
        .compactMap({ prediction.featureValue(for: $0)?.imageBufferValue })
        .first
      else {
        print("[LightPipeline] depth: prediction has no image output")
        return
      }
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

    lock.lock()
    let backIndex = 1 - frontIndex
    lock.unlock()
    let target = buffers[backIndex]

    CVPixelBufferLockBaseAddress(output, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(output, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(output) else { return }
    let bytesPerRow = CVPixelBufferGetBytesPerRow(output)
    let format = CVPixelBufferGetPixelFormatType(output)

    switch format {
    case kCVPixelFormatType_OneComponent16Half, kCVPixelFormatType_DepthFloat16:
      var src = vImage_Buffer(
        data: UnsafeMutableRawPointer(mutating: base),
        height: vImagePixelCount(outHeight), width: vImagePixelCount(outWidth),
        rowBytes: bytesPerRow)
      var dst = vImage_Buffer(
        data: UnsafeMutableRawPointer(target),
        height: vImagePixelCount(outHeight), width: vImagePixelCount(outWidth),
        rowBytes: outWidth * MemoryLayout<Float>.stride)
      vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
    case kCVPixelFormatType_OneComponent32Float, kCVPixelFormatType_DepthFloat32:
      for y in 0..<outHeight {
        let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float.self)
        target.advanced(by: y * outWidth).update(from: row, count: outWidth)
      }
    default:
      print("[LightPipeline] depth: unsupported output format \(format)")
      return
    }

    let (low, high) = Self.robustRange(of: target, count: outWidth * outHeight)
    let elapsed = (CACurrentMediaTime() - start) * 1000

    lock.lock()
    frontIndex = backIndex
    depthSeq += 1
    depthLow = low
    depthHigh = high
    depthTimeMs = elapsed
    depthPrepMs = (prepDone - start) * 1000
    depthPredictMs = (predictDone - prepDone) * 1000
    lock.unlock()
  }

  /// 2nd/98th percentile of `values` via a 256-bin histogram (the same robust
  /// normalization the TypeGPU demo computes on the GPU).
  private static func robustRange(of values: UnsafeMutablePointer<Float>, count: Int)
    -> (Float, Float) {
    var minV: Float = 0
    var maxV: Float = 0
    vDSP_minv(values, 1, &minV, vDSP_Length(count))
    vDSP_maxv(values, 1, &maxV, vDSP_Length(count))
    guard minV.isFinite, maxV.isFinite, maxV > minV else { return (0, 1) }
    let span = maxV - minV
    var histogram = [Int](repeating: 0, count: 256)
    let scale = 255.0 / span
    for i in 0..<count {
      let v = values[i]
      if v.isFinite {
        let bin = Int((v - minV) * scale)
        histogram[min(max(bin, 0), 255)] += 1
      }
    }
    let total = histogram.reduce(0, +)
    guard total > 0 else { return (0, 1) }
    let lowTarget = max(1, Int(Double(total) * 0.02))
    let highTarget = Int(Double(total) * 0.98)
    var cumulative = 0
    var lowBin = 0
    var highBin = 255
    var lowFound = false
    for bin in 0..<256 {
      cumulative += histogram[bin]
      if !lowFound, cumulative >= lowTarget {
        lowBin = bin
        lowFound = true
      }
      if cumulative >= highTarget {
        highBin = bin
        break
      }
    }
    let low = minV + (Float(lowBin) / 256.0) * span
    var high = minV + (Float(highBin + 1) / 256.0) * span
    high = max(high, low + 0.001)
    return (low, high)
  }

  func getDepthResult() throws -> DepthResult {
    lock.lock()
    let seq = depthSeq
    let index = frontIndex
    let low = depthLow
    let high = depthHigh
    let time = depthTimeMs
    let prep = depthPrepMs
    let predict = depthPredictMs
    lock.unlock()
    let count = modelWidth * modelHeight
    let pointer = buffers[index]
    // Zero-copy view; keep `self` alive while JS holds the buffer.
    let buffer = ArrayBuffer.wrap(
      dataWithoutCopy: UnsafeMutableRawPointer(pointer),
      size: count * MemoryLayout<Float>.stride,
      onDelete: { _ = self })
    return DepthResult(
      seq: Double(seq), width: Double(modelWidth), height: Double(modelHeight),
      low: Double(low), high: Double(high), inferenceTimeMs: time,
      prepTimeMs: prep, predictTimeMs: predict, data: buffer)
  }

  // MARK: - Hands

  /// Measure the in-plane rotation of a face in the RAW buffer. Vision's
  /// face detector is rotation-invariant, so the observation's `roll` angle
  /// tells us directly how the buffer content is rotated.
  private func scanOrientation(on pixelBuffer: CVPixelBuffer) {
    let request = VNDetectFaceRectanglesRequest()
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
    try? handler.perform([request])
    guard let face = request.results?.max(by: { $0.confidence < $1.confidence }),
          face.confidence > 0.5,
          let roll = face.roll?.doubleValue
    else { return }
    lock.lock()
    lastRollDeg = roll * 180 / .pi
    lock.unlock()
  }

  private func runHandDetection(
    on pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation
  ) {
    // Piggyback the orientation auto-calibration on this queue: measure
    // often until a face has been seen, then re-measure every ~3s (gimbal
    // cameras can physically rotate mid-session).
    orientationScanCountdown -= 1
    if orientationScanCountdown <= 0 {
      scanOrientation(on: pixelBuffer)
      lock.lock()
      let seen = lastRollDeg > -900
      lock.unlock()
      orientationScanCountdown = seen ? 120 : 30
    }

    let start = CACurrentMediaTime()
    let request = VNDetectHumanHandPoseRequest()
    request.maximumHandCount = 1
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation)

    // Vision reports landmarks in the ORIENTED (upright) image space, so the
    // crop math uses upright dimensions (swapped for 90/270 rotations).
    let rotated = orientation == .right || orientation == .left
      || orientation == .rightMirrored || orientation == .leftMirrored
    let bufferWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
    let bufferHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
    let frameWidth = rotated ? bufferHeight : bufferWidth
    let frameHeight = rotated ? bufferWidth : bufferHeight

    var detected: (
      thumb: (Double, Double), index: (Double, Double),
      pinchRatio: Double, confidence: Double
    )? = nil

    do {
      try handler.perform([request])
    } catch {
      // fall through with detected == nil
    }
    if let hand = request.results?.first,
       let thumb = try? hand.recognizedPoint(.thumbTip),
       let index = try? hand.recognizedPoint(.indexTip),
       let wrist = try? hand.recognizedPoint(.wrist),
       let middleMCP = try? hand.recognizedPoint(.middleMCP),
       thumb.confidence > 0.3, index.confidence > 0.3 {
      // Vision: normalized [0,1], origin BOTTOM-LEFT of the full frame.
      // Convert into the center-cropped (model aspect) region, origin TOP-LEFT.
      let modelAspect = CGFloat(modelWidth) / CGFloat(modelHeight)
      let frameAspect = frameWidth / frameHeight
      var cropX: CGFloat = 0
      var cropY: CGFloat = 0
      var cropW: CGFloat = 1
      var cropH: CGFloat = 1
      if frameAspect > modelAspect {
        cropW = modelAspect / frameAspect
        cropX = (1 - cropW) / 2
      } else {
        cropH = frameAspect / modelAspect
        cropY = (1 - cropH) / 2
      }
      func toCropSpace(_ p: VNRecognizedPoint) -> (Double, Double) {
        let x = (p.location.x - cropX) / cropW
        let y = ((1 - p.location.y) - cropY) / cropH
        return (Double(x), Double(y))
      }
      let handSize = max(
        hypot(wrist.location.x - middleMCP.location.x,
              wrist.location.y - middleMCP.location.y),
        1e-4)
      let pinchDistance = hypot(thumb.location.x - index.location.x,
                                thumb.location.y - index.location.y)
      detected = (
        thumb: toCropSpace(thumb),
        index: toCropSpace(index),
        pinchRatio: Double(pinchDistance / handSize),
        confidence: Double(min(thumb.confidence, index.confidence))
      )
    }

    let elapsed = (CACurrentMediaTime() - start) * 1000
    lock.lock()
    handSeq += 1
    if let hand = detected {
      latestHand = HandResult(
        seq: Double(handSeq), tracked: true,
        thumbX: hand.thumb.0, thumbY: hand.thumb.1,
        indexX: hand.index.0, indexY: hand.index.1,
        midX: (hand.thumb.0 + hand.index.0) / 2,
        midY: (hand.thumb.1 + hand.index.1) / 2,
        pinchRatio: hand.pinchRatio, confidence: hand.confidence,
        detectionTimeMs: elapsed)
    } else {
      latestHand = HandResult(
        seq: Double(handSeq), tracked: false, thumbX: 0, thumbY: 0,
        indexX: 0, indexY: 0, midX: 0, midY: 0, pinchRatio: 1,
        confidence: 0, detectionTimeMs: elapsed)
    }
    lock.unlock()
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

/// Minimal atomic test-and-set flag (drop-if-busy gate).
final class NitroAtomicFlag {
  private let lock = NSLock()
  private var value = false

  /// Returns the previous value and sets the flag.
  func testAndSet() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    let previous = value
    value = true
    return previous
  }

  func clear() {
    lock.lock()
    value = false
    lock.unlock()
  }
}
