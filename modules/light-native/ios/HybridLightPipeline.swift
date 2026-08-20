//
//  HybridLightPipeline.swift
//  LightDemo
//

import Accelerate
import CoreML
import CoreVideo
import Foundation
import NitroModules
import Vision

/// Frame analysis pipeline: CoreML monocular depth (ANE) + Vision hand pose.
/// Both tasks run on their own serial queues with drop-if-busy semantics.
final class HybridLightPipeline: HybridLightPipelineSpec {
  private let model: VNCoreMLModel
  private let modelWidth: Int
  private let modelHeight: Int

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

  private var handSeq: Int = -1
  private var latestHand: HandResult

  private var controls: LightControls
  private var status: LightStatus

  var depthWidth: Double { Double(modelWidth) }
  var depthHeight: Double { Double(modelHeight) }

  init(model: MLModel, inputWidth: Int, inputHeight: Int) throws {
    self.model = try VNCoreMLModel(for: model)
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
      handControl: true, snapshotPath: "")
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

  func submitFrame(pointer: UInt64, runDepth: Bool, runHands: Bool) throws {
    guard let raw = UnsafeRawPointer(bitPattern: UInt(pointer)) else {
      throw RuntimeError.error(withMessage: "submitFrame: pointer is null")
    }
    let pixelBuffer = Unmanaged<CVPixelBuffer>.fromOpaque(raw).takeUnretainedValue()

    if runDepth, !depthBusy.testAndSet() {
      // Retain the buffer for the async task; released when done.
      let retained = Unmanaged<CVPixelBuffer>.fromOpaque(raw).retain().takeUnretainedValue()
      depthQueue.async { [weak self] in
        defer {
          Unmanaged.passUnretained(retained).release()
          self?.depthBusy.clear()
        }
        self?.runDepthInference(on: retained)
      }
    }
    if runHands, !handBusy.testAndSet() {
      let retained = Unmanaged<CVPixelBuffer>.fromOpaque(raw).retain().takeUnretainedValue()
      handQueue.async { [weak self] in
        defer {
          Unmanaged.passUnretained(retained).release()
          self?.handBusy.clear()
        }
        self?.runHandDetection(on: retained)
      }
    }
    _ = pixelBuffer // caller keeps its own +1 until we return
  }

  // MARK: - Depth

  private func runDepthInference(on pixelBuffer: CVPixelBuffer) {
    let start = CACurrentMediaTime()
    let request = VNCoreMLRequest(model: model)
    request.imageCropAndScaleOption = .centerCrop
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
    do {
      try handler.perform([request])
    } catch {
      print("[LightPipeline] depth inference failed: \(error)")
      return
    }
    guard let observation = request.results?.first as? VNPixelBufferObservation else {
      print("[LightPipeline] depth: no VNPixelBufferObservation result")
      return
    }
    let output = observation.pixelBuffer
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
      low: Double(low), high: Double(high), inferenceTimeMs: time, data: buffer)
  }

  // MARK: - Hands

  private func runHandDetection(on pixelBuffer: CVPixelBuffer) {
    let start = CACurrentMediaTime()
    let request = VNDetectHumanHandPoseRequest()
    request.maximumHandCount = 1
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)

    let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
    let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

    var result = HandResult(
      seq: 0, tracked: false, thumbX: 0, thumbY: 0, indexX: 0, indexY: 0,
      midX: 0, midY: 0, pinchRatio: 1, confidence: 0, detectionTimeMs: 0)

    defer {
      result.detectionTimeMs = (CACurrentMediaTime() - start) * 1000
      lock.lock()
      handSeq += 1
      result.seq = Double(handSeq)
      latestHand = result
      lock.unlock()
    }

    do {
      try handler.perform([request])
    } catch {
      return
    }
    guard let hand = request.results?.first,
          let thumb = try? hand.recognizedPoint(.thumbTip),
          let index = try? hand.recognizedPoint(.indexTip),
          let wrist = try? hand.recognizedPoint(.wrist),
          let middleMCP = try? hand.recognizedPoint(.middleMCP),
          thumb.confidence > 0.3, index.confidence > 0.3
    else {
      return
    }

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

    let (tx, ty) = toCropSpace(thumb)
    let (ix, iy) = toCropSpace(index)
    let handSize = max(
      hypot(wrist.location.x - middleMCP.location.x,
            wrist.location.y - middleMCP.location.y),
      1e-4)
    let pinchDistance = hypot(thumb.location.x - index.location.x,
                              thumb.location.y - index.location.y)

    result.tracked = true
    result.thumbX = tx
    result.thumbY = ty
    result.indexX = ix
    result.indexY = iy
    result.midX = (tx + ix) / 2
    result.midY = (ty + iy) / 2
    result.pinchRatio = Double(pinchDistance / handSize)
    result.confidence = Double(min(thumb.confidence, index.confidence))
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
