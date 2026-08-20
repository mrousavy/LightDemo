import type { HybridObject } from 'react-native-nitro-modules'
import type { LightPipeline } from './LightPipeline.nitro'

/**
 * Root factory for the LightDemo native pipeline, plus debug utilities.
 */
export interface LightNative
  extends HybridObject<{ ios: 'swift' }> {
  /**
   * Load + compile (cached) the Depth Anything V2 CoreML model, warm it up
   * on the Neural Engine, and return a ready {@linkcode LightPipeline}.
   */
  createPipeline(): Promise<LightPipeline>

  /**
   * Debug: snapshot the app's key window (including Metal/WebGPU layers)
   * to a PNG at {@linkcode path}. Resolves with true on success.
   */
  snapshotWindow(path: string): Promise<boolean>

  /**
   * Debug: write raw pixels to a PNG at {@linkcode path}.
   * {@linkcode bgra} selects BGRA byte order (WebGPU bgra8unorm readbacks),
   * otherwise RGBA is assumed.
   */
  savePng(
    path: string,
    width: number,
    height: number,
    bytesPerRow: number,
    bgra: boolean,
    data: ArrayBuffer
  ): boolean
}
