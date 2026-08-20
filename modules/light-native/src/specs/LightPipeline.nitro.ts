import type { HybridObject } from 'react-native-nitro-modules'
import type { Frame } from 'react-native-vision-camera'
import type { DepthResult } from '../types/DepthResult'
import type { HandResult } from '../types/HandResult'
import type { LightControls } from '../types/LightControls'
import type { LightStatus } from '../types/LightStatus'

/**
 * A ready-to-use frame analysis pipeline:
 * - Monocular depth estimation (Depth Anything V2 Small, CoreML on the ANE)
 * - Hand pose / pinch detection (Apple Vision framework)
 *
 * Both run asynchronously on their own background queues with
 * drop-if-busy semantics; results are polled with the getters.
 *
 * Created via {@linkcode LightNative.createPipeline | createPipeline()},
 * which loads + warms up the CoreML model.
 */
export interface LightPipeline
  extends HybridObject<{ ios: 'swift' }> {
  /** Width of the depth model's input/output in pixels. */
  readonly depthWidth: number
  /** Height of the depth model's input/output in pixels. */
  readonly depthHeight: number

  /**
   * Analyze one camera frame FULLY SYNCHRONOUSLY on the calling thread:
   * depth inference (~20-30ms on the ANE), and - when {@linkcode runHands} -
   * hand-pose detection, both for THIS exact frame. Rendering with the same
   * frame's results keeps camera pixels and lighting perfectly coherent (an
   * asynchronously-lagging depth map paints glowing ghost trails behind
   * fast-moving objects).
   *
   * Hand detection runs on the already-prepared (GPU crop/scaled, upright)
   * depth-model input buffer, so its coordinates come back directly in the
   * crop space shared with the depth map.
   *
   * Takes the VisionCamera {@linkcode Frame} directly (typed, first-class -
   * this library declares its react-native-vision-camera dependency; the
   * untyped NativeBuffer pointer contract is only for dependency-free
   * consumers like react-native-webgpu). {@linkcode orientationDegrees} is
   * the rotation needed to display the buffer upright (up=0, right=90,
   * down=180, left=270).
   */
  analyzeSync(
    frame: Frame,
    orientationDegrees: number,
    runHands: boolean
  ): DepthResult

  /** Latest completed depth inference (seq -1 if none yet). */
  getDepthResult(): DepthResult
  /** Latest completed hand detection (seq -1 if none yet). */
  getHandResult(): HandResult

  /**
   * Max disparity over small neighborhoods around the given crop-space
   * points (flat `[x0, y0, x1, y1, ...]`, normalized 0-1), read from the
   * latest depth map. A bounded ~tens-of-texels probe for placing the light
   * at the fingertips' depth. Returns -1 when no depth is available.
   */
  sampleDepthMax(points: number[]): number

  /**
   * Cross-runtime parameter store: the React runtime writes UI parameters,
   * the frame-processor worklet reads them each frame. Thread-safe.
   */
  setControls(controls: LightControls): void
  getControls(): LightControls

  /**
   * Cross-runtime status store: the frame-processor worklet writes live
   * stats, the React runtime polls them for the HUD. Thread-safe.
   */
  setStatus(status: LightStatus): void
  getStatus(): LightStatus
}
