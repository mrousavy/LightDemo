// Hermes provides `performance` at runtime (also inside worklet runtimes),
// but the narrowed tsconfig "types" list drops its declaration.
declare const performance: { now(): number }

// LIGHTDEMO PATCH (react-native-webgpu): immutable pre-resolved dispatch
// list, built once via device.createDispatchList() and replayed with a
// single JSI call via pass.executeDispatchList().
interface GPUDispatchList {
  readonly __brand: 'GPUDispatchList'
  readonly count: number
}
interface GPUDevice {
  createDispatchList(
    items: Array<{
      pipeline: GPUComputePipeline
      bindGroup: GPUBindGroup
      x: number
      y?: number
      z?: number
    }>,
  ): GPUDispatchList
}
interface GPUComputePassEncoder {
  executeDispatchList(list: GPUDispatchList): void
}
