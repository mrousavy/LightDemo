//
//  HybridLightNative.swift
//  LightDemo
//

import CoreML
import Foundation
import ImageIO
import NitroModules
import UIKit
import UniformTypeIdentifiers

/// Root factory: loads the Depth Anything V2 CoreML model and creates
/// ready-to-use pipelines. Also hosts small debug utilities.
final class HybridLightNative: HybridLightNativeSpec {
  private static let setupQueue = DispatchQueue(label: "light.setup", qos: .userInitiated)

  func createPipeline() throws -> Promise<any HybridLightPipelineSpec> {
    return Promise.parallel(Self.setupQueue) {
      let model = try Self.loadDepthModel()
      let (width, height) = try Self.imageInputSize(of: model)
      let pipeline = try HybridLightPipeline(
        model: model, inputWidth: width, inputHeight: height)
      // Warm up the ANE with one dummy prediction so the first camera frame
      // doesn't pay the lazy-compilation cost.
      Self.warmUp(model: model, width: width, height: height)
      return pipeline
    }
  }

  private static func loadDepthModel() throws -> MLModel {
    let configuration = MLModelConfiguration()
    // .all lets CoreML's cost model pick the GPU in this iOS-on-Mac process
    // (~42ms/frame); pinning to the Neural Engine measures ~17ms on M1 Max.
    if #available(iOS 16.0, *) {
      configuration.computeUnits = .cpuAndNeuralEngine
    } else {
      configuration.computeUnits = .all
    }

    let bundle = modelsBundle()
    // Preferred: the resource-bundle target already compiled the .mlpackage.
    if let compiledURL = bundle?.url(forResource: "DepthAnythingV2Small", withExtension: "mlmodelc") {
      return try MLModel(contentsOf: compiledURL, configuration: configuration)
    }
    // Fallback: raw .mlpackage was only copied - compile once and cache.
    guard let packageURL = bundle?.url(forResource: "DepthAnythingV2Small", withExtension: "mlpackage") else {
      throw RuntimeError.error(withMessage:
        "DepthAnythingV2Small model not found in LightNativeModels bundle")
    }
    let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
    let cachedURL = caches.appendingPathComponent("DepthAnythingV2Small.mlmodelc")
    if FileManager.default.fileExists(atPath: cachedURL.path) {
      return try MLModel(contentsOf: cachedURL, configuration: configuration)
    }
    let compiledURL = try MLModel.compileModel(at: packageURL)
    try? FileManager.default.removeItem(at: cachedURL)
    try FileManager.default.copyItem(at: compiledURL, to: cachedURL)
    return try MLModel(contentsOf: cachedURL, configuration: configuration)
  }

  private static func modelsBundle() -> Bundle? {
    let container = Bundle(for: HybridLightNative.self)
    if let url = container.url(forResource: "LightNativeModels", withExtension: "bundle"),
       let bundle = Bundle(url: url) {
      return bundle
    }
    if let url = Bundle.main.url(forResource: "LightNativeModels", withExtension: "bundle"),
       let bundle = Bundle(url: url) {
      return bundle
    }
    return nil
  }

  private static func imageInputSize(of model: MLModel) throws -> (Int, Int) {
    for (_, description) in model.modelDescription.inputDescriptionsByName {
      if let constraint = description.imageConstraint {
        return (constraint.pixelsWide, constraint.pixelsHigh)
      }
    }
    throw RuntimeError.error(withMessage: "Depth model has no image input")
  }

  private static func warmUp(model: MLModel, width: Int, height: Int) {
    var pixelBuffer: CVPixelBuffer?
    let attributes: [CFString: Any] = [
      kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
    ]
    CVPixelBufferCreate(
      kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
      attributes as CFDictionary, &pixelBuffer)
    guard let buffer = pixelBuffer else { return }
    guard let inputName = model.modelDescription.inputDescriptionsByName.first(where: {
      $0.value.imageConstraint != nil
    })?.key else { return }
    let value = MLFeatureValue(pixelBuffer: buffer)
    if let provider = try? MLDictionaryFeatureProvider(dictionary: [inputName: value]) {
      _ = try? model.prediction(from: provider)
    }
  }

  // MARK: - Debug utilities

  func snapshotWindow(path: String) throws -> Promise<Bool> {
    // Relative paths resolve into the app's Documents directory (readable
    // from the Mac at ~/Library/Containers/<bundle-id>/Data/Documents).
    let resolved: String
    if path.hasPrefix("/") {
      resolved = path
    } else {
      let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
      resolved = documents.appendingPathComponent(path).path
    }
    return Promise<Bool>.async { @MainActor in
      guard let window = UIApplication.shared.connectedScenes
        .compactMap({ $0 as? UIWindowScene })
        .flatMap({ $0.windows })
        .first(where: { $0.isKeyWindow }) ?? UIApplication.shared.windows.first
      else {
        return false
      }
      let renderer = UIGraphicsImageRenderer(bounds: window.bounds)
      let image = renderer.image { _ in
        window.drawHierarchy(in: window.bounds, afterScreenUpdates: true)
      }
      guard let data = image.pngData() else { return false }
      try data.write(to: URL(fileURLWithPath: resolved))
      return true
    }
  }

  func savePng(
    path: String, width: Double, height: Double, bytesPerRow: Double,
    bgra: Bool, data: ArrayBuffer
  ) throws -> Bool {
    let w = Int(width)
    let h = Int(height)
    let stride = Int(bytesPerRow)
    let bytes = data.toData(copyIfNeeded: true)
    guard bytes.count >= stride * h else {
      throw RuntimeError.error(withMessage:
        "savePng: buffer too small (\(bytes.count) < \(stride * h))")
    }
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    var bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue
      | CGImageAlphaInfo.noneSkipFirst.rawValue
    if !bgra {
      bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue
        | CGImageAlphaInfo.noneSkipLast.rawValue
    }
    guard let provider = CGDataProvider(data: bytes as CFData),
          let image = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: stride, space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent)
    else {
      return false
    }
    let url = URL(fileURLWithPath: path) as CFURL
    guard let destination = CGImageDestinationCreateWithURL(
      url, UTType.png.identifier as CFString, 1, nil)
    else {
      return false
    }
    CGImageDestinationAddImage(destination, image, nil)
    return CGImageDestinationFinalize(destination)
  }
}
