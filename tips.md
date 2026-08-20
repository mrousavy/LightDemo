It can still be optimized. The overall architecture is already quite strong—persistent buffers, direct Core ML instead of `VNCoreMLRequest`, IOSurface-backed output, one reusable `CIContext`, small Vision input—but I would not call it the fastest possible implementation yet.

The biggest opportunities, roughly in order:

1. **Run Core ML depth + Vision hand pose concurrently.**

   Right now the pipeline is:

   ```text
   CoreImage preprocess
        ↓
   CoreML depth ~16.6 ms
        ↓
   Vision hands
        ↓
   return
   ```

   because `runDepthInference()` completes before `detectHands()` starts.

   Once `prepareModelInput()` has produced your 392×294 buffer, both consumers are independent. So ideally:

   ```text
                     ┌── CoreML depth ───────┐
   preprocess ───────┤                       ├── join
                     └── Vision hand pose ───┘
   ```

   If depth = 16.6 ms and hands =, say, 8 ms, serial latency is ~24.6 ms. The theoretical concurrent latency is closer to ~16.6 ms.

   There is one caveat: Vision and your Core ML model may compete for ANE/GPU resources, so this needs benchmarking. Apple also recommends using a particular `MLModel` instance from one thread/queue at a time, but your hand detector is separate from that model, so that restriction doesn't prevent this architecture. ([Apple Developer][1])

2. **Stop creating an `MLDictionaryFeatureProvider` every frame.**

   You're doing this every prediction:

   ```swift
   let provider = try MLDictionaryFeatureProvider(
     dictionary: [modelInputName: MLFeatureValue(pixelBuffer: input)]
   )
   ```



   But `input` is always the same `CVPixelBuffer` object. Only its contents change.

   Allocate the input buffer eagerly and create these once:

   ```swift
   private let inputBuffer: CVPixelBuffer
   private let inputFeature: MLFeatureValue
   private let inputProvider: MLDictionaryFeatureProvider
   ```

   then every frame:

   ```swift
   let prediction = try mlModel.prediction(
     from: inputProvider,
     options: predictionOptions
   )
   ```

   `MLFeatureValue(pixelBuffer:)` holds the pixel buffer, so Core ML sees its updated contents. Apple's documented Core ML input mechanism is directly based on `CVPixelBuffer`. ([Apple Developer][2])

   This won't save milliseconds, but there's no reason for the per-frame dictionaries/NSObject allocation.

3. **Your `sparseRobustRange()` is more suspicious than it looks.**

   You're doing:

   ```swift
   CVPixelBufferLockBaseAddress(output, .readOnly)
   ...
   // read 1024 Float16s
   samples.sort()
   ```

   every frame.

   The 1024 reads and sort aren't particularly expensive. The concern is this:

   ```text
   ANE/GPU writes IOSurface
             ↓
   CPU locks/touches IOSurface
             ↓
   WebGPU imports/reads IOSurface
   ```

   You're deliberately building a GPU/ANE→WebGPU zero-copy path, then touching that same surface from the CPU every frame.

   Since `prediction()` is synchronous, some synchronization already necessarily occurred, so I wouldn't claim the lock is definitely costing multiple milliseconds. But I would absolutely benchmark:

   ```swift
   // Completely disable:
   let (low, high) = (previousLow, previousHigh)
   ```

   If frame time changes materially, move range calculation to:

   * Metal/WebGPU, or
   * every 5–10 frames with interpolation/EMA.

   For a slowly changing depth normalization range, calculating percentiles at 30/60 Hz is probably unnecessary.

4. **Your Core Image context is explicitly configured for lower memory, not maximum speed.**

   You have:

   ```swift
   CIContext(options: [.cacheIntermediates: false])
   ```



   Apple explicitly says caching intermediates can make subsequent similar renders faster, while `false` reduces memory usage. ([Apple Developer][3])

   I'd first try:

   ```swift
   private let ciContext = CIContext()
   ```

   Your graph is simple enough that the difference might be tiny because Core Image can fuse the affine/crop operations anyway, but `false` is definitely not the "optimize for maximum speed" setting.

5. **Disable Core Image color management if your model doesn't need it.**

   The default Core Image context performs automatic color management and uses extended linear sRGB as its working color space. Apple lets you pass `NSNull()` to disable it. ([Apple Developer][4])

   For ML preprocessing, I'd benchmark:

   ```swift
   CIContext(options: [
     .workingColorSpace: NSNull(),
     .cacheIntermediates: true
   ])
   ```

   This can remove unnecessary color-space work.

   **But validate model accuracy.** The numerical RGB values presented to the model can change depending on how your original model preprocessing was defined.

6. **Keep low-quality single-pass downscaling.**

   You're dramatically reducing camera resolution down to 392×294. On iOS, Core Image defaults to the faster single-pass downsample path. Do **not** enable `.highQualityDownsample`. Apple says high-quality downsampling can use multiple passes, whereas low quality performs it in one pass for better performance. ([Apple Developer][5])

7. **If preprocessing itself is significant, replace Core Image with one Metal kernel.**

   Your current preprocessing is actually quite sensible:

   ```swift
   CIImage(pixelBuffer)
       → orientation
       → crop
       → translation
       → scale
       → render(BGRA CVPixelBuffer)
   ```



   Core Image is lazy, so these aren't necessarily four separate GPU passes.

   But the absolute fastest implementation would likely be one custom Metal compute kernel that directly does:

   ```text
   Camera YUV/BGRA
      ↓
   rotate + crop + sample + YUV→RGB
      ↓
   392×294 BGRA IOSurface
   ```

   in a single dispatch.

   Whether it's worth it depends entirely on your measured `depthPrepMs`. If preparation is:

   * **0.3–1 ms:** don't bother.
   * **2–4 ms:** Metal becomes interesting.
   * **5+ ms:** I'd definitely replace CI.

8. **Core ML model loading/configuration still has some knobs.**

   If you control how `mlModel` is loaded, newer Core ML has optimization hints including `.fastPrediction`, which explicitly trades model specialization/load costs and storage for lower prediction latency. ([Apple Developer][6])

   And if your model has flexible shapes but you always use 392×294, tell Core ML that shape changes are `.infrequent`; Apple says this permits more shape-specific optimization for faster subsequent predictions. ([Apple Developer][7])

   I'd also benchmark:

   ```swift
   config.computeUnits = .all
   ```

   against:

   ```swift
   config.computeUnits = .cpuAndNeuralEngine
   ```

   `.all` lets Core ML choose across CPU/GPU/ANE and is generally the right default. `.cpuAndNeuralEngine` can nevertheless help in an app where the GPU is already busy doing Core Image + WebGPU rendering, because it explicitly prevents Core ML from putting graph segments on the GPU. ([Apple Developer][8])

9. **The lock stuff isn't worth optimizing yet.**

   You currently have a fair number of:

   ```swift
   lock.lock()
   ...
   lock.unlock()
   ```

   including storing the depth result and then immediately reacquiring the same lock through `getDepthResult()`.

   You could return the locally generated `DepthResult` directly and use `OSAllocatedUnfairLock` or split the locks by state domain.

   But we're talking microseconds/nanoseconds here. Don't touch this before measuring items 1–5.

10. **Face orientation scanning is a latency spike rather than steady-state cost.**

Every ~120 frames you run a new face detection over the raw camera frame:

```swift
let request = VNDetectFaceRectanglesRequest()
let handler = VNImageRequestHandler(
  cvPixelBuffer: pixelBuffer,
  orientation: .up
)
try? handler.perform([request])
```



Average cost is low because it's infrequent, but your frame-time graph will probably contain a periodic spike. If smoothness matters, I'd perform this asynchronously and consume the latest orientation estimate on subsequent frames. It doesn't need to correspond exactly to the same rendered frame.

### What I would actually change

My version would conceptually become:

```text
camera frame
   │
   ▼
ONE reusable preprocessing target
Metal/CI: rotate + crop + resize
   │
   ├───────────────────────┐
   ▼                       ▼
CoreML depth             Vision hands
ANE                      Vision executor
   │                       │
   └───────── join ─────────┘
             │
             ▼
IOSurface depth
             │
             ├── WebGPU directly
             │
             └── range calculation only occasionally / GPU
```

And I'd preallocate **everything**:

* input `CVPixelBuffer`
* output `CVPixelBuffer`
* `MLFeatureValue`
* `MLFeatureProvider`
* `MLPredictionOptions`
* Vision request
* sequence handler

You're already doing most of that.

### My ranking

| Change                                                | Expected value                 |
| ----------------------------------------------------- | ------------------------------ |
| Parallelize hand + depth                              | **Potentially huge**           |
| Remove per-frame CPU depth read/range                 | **Potentially significant**    |
| Core ML `.fastPrediction` / compute-unit benchmarking | **Potentially significant**    |
| Metal preprocessing instead of CI                     | **Potentially 0.5–several ms** |
| Disable CI color management                           | **Potentially useful**         |
| Reuse feature provider                                | Small but essentially free     |
| Enable CI intermediate caching                        | Small/benchmark                |
| Optimize NSLock / Swift math                          | Negligible                     |

So: **the direct Core ML + reusable IOSurface approach is fundamentally the right fast path.** I would not switch back to Vision/CoreML integration; your measured 40 ms vs 16.6 ms already makes that clear. The remaining major wins are in **scheduling/parallelism and avoiding CPU synchronization**, not finding some secret faster Core ML call.

If your current logged `depthPrepMs`, `depthPredictMs`, and `handTimeMs` are available, those three numbers would make it possible to say very precisely where the next milliseconds are.

[1]: https://developer.apple.com/documentation/coreml/mlmodel?changes=_2&utm_source=chatgpt.com "MLModel | Apple Developer Documentation"
[2]: https://developer.apple.com/documentation/coreml/mlfeaturevalue/init%28pixelbuffer%3A%29?utm_source=chatgpt.com "init(pixelBuffer:) | Apple Developer Documentation"
[3]: https://developer.apple.com/documentation/coreimage/cicontextoption/cacheintermediates?changes=_6&language=objc&utm_source=chatgpt.com "kCIContextCacheIntermediates | Apple Developer Documentation"
[4]: https://developer.apple.com/documentation/coreimage/cicontextoption/workingcolorspace?utm_source=chatgpt.com "workingColorSpace | Apple Developer Documentation"
[5]: https://developer.apple.com/documentation/coreimage/cicontextoption/highqualitydownsample?language=objc&utm_source=chatgpt.com "kCIContextHighQualityDownsample | Apple Developer Documentation"
[6]: https://developer.apple.com/documentation/coreml/mlspecializationstrategy?changes=_4&utm_source=chatgpt.com "MLSpecializationStrategy | Apple Developer Documentation"
[7]: https://developer.apple.com/documentation/coreml/mloptimizationhints-swift.struct/reshapefrequency-swift.enum?utm_source=chatgpt.com "MLOptimizationHints.ReshapeFrequency | Apple Developer Documentation"
[8]: https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine?changes=_8&utm_source=chatgpt.com "MLComputeUnits.cpuAndNeuralEngine | Apple Developer Documentation"
