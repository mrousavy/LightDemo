import type { HybridObject, UInt64 } from 'react-native-nitro-modules'
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
   * The in-plane rotation (roll, degrees) of the most recently detected face
   * in the RAW buffer, measured periodically on the hand-detection queue, or
   * -999 while no face has been seen. Modern Vision face detection is
   * rotation-invariant, so a "which orientation finds a face" scan cannot
   * discriminate - but the observation's roll angle directly tells us how
   * the buffer is rotated. The JS side maps this to a display rotation
   * (gimbal cameras can physically rotate mid-session, so it re-measures
   * continuously).
   */
  readonly lastFaceRollDegrees: number

  /**
   * Submit a camera frame (a `CVPixelBufferRef` pointer from
   * VisionCamera's `frame.getNativeBuffer().pointer`, retained +1 by the
   * caller) for analysis. The pipeline retains the buffer internally for as
   * long as it needs it - the caller can `release()` its reference
   * immediately after this returns.
   *
   * Frames submitted while a task is still busy are dropped for that task.
   * Safe to call from the VisionCamera frame-processor worklet thread.
   *
   * {@linkcode orientationDegrees} is the rotation needed to display the
   * buffer upright (VisionCamera's `Frame.orientation`: up=0, right=90,
   * down=180, left=270). Depth maps and hand coordinates are produced in
   * the upright (display) space.
   */
  submitFrame(
    pointer: UInt64,
    orientationDegrees: number,
    runDepth: boolean,
    runHands: boolean
  ): void

  /** Latest completed depth inference (seq -1 if none yet). */
  getDepthResult(): DepthResult

  /**
   * Run depth inference for this frame SYNCHRONOUSLY on the calling thread
   * (~20-30ms on the ANE) and return the result. Rendering with the same
   * frame's depth keeps camera pixels and lighting perfectly coherent - an
   * asynchronously-lagging depth map paints glowing ghost trails behind
   * fast-moving objects. Do not mix with `submitFrame(runDepth: true)`.
   */
  runDepthSync(pointer: UInt64, orientationDegrees: number): DepthResult
  /** Latest completed hand detection (seq -1 if none yet). */
  getHandResult(): HandResult

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
