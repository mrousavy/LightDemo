import type { UInt64 } from 'react-native-nitro-modules'

/**
 * One completed monocular depth inference.
 *
 * The depth map itself never leaves the GPU: {@linkcode surfacePointer} is
 * the `IOSurfaceRef` of CoreML's output pixel buffer (single-channel f16,
 * row-major, y-down, crop space; larger = closer / relative disparity).
 * Import it into WebGPU via `device.importSharedTextureMemory({ handle })`
 * as an `r16float` texture. The surface stays valid until the next
 * `analyzeSync` call replaces it - import and encode within the same frame.
 */
export interface DepthResult {
  /** Monotonically increasing sequence number. -1 if no inference completed yet. */
  seq: number
  /** Width of the depth map in pixels. */
  width: number
  /** Height of the depth map in pixels. */
  height: number
  /**
   * Robust disparity range of this frame (2nd/98th percentile), estimated
   * from a sparse 1024-sample grid probe - a bounded statistical reduction,
   * not a full-frame CPU read.
   */
  low: number
  high: number
  /** Inference wall time in milliseconds (for the HUD). */
  inferenceTimeMs: number
  /** Preprocessing (crop/scale) part of {@linkcode inferenceTimeMs}. */
  prepTimeMs: number
  /** Pure CoreML prediction part of {@linkcode inferenceTimeMs}. */
  predictTimeMs: number
  /** IOSurfaceRef of the f16 depth map (see interface docs). */
  surfacePointer: UInt64
}
