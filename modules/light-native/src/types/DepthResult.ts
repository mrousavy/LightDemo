/**
 * One completed monocular depth inference.
 *
 * {@linkcode data} is a zero-copy view into a native ping-pong buffer of
 * `width * height` Float32 values (row-major, y-down, in the coordinate
 * space of the model's center-cropped input). Larger values = closer
 * (relative inverse depth / disparity). The buffer stays valid until the
 * pipeline completes two more inferences, so consume (upload) it right away.
 */
export interface DepthResult {
  /** Monotonically increasing sequence number. -1 if no inference completed yet. */
  seq: number
  /** Width of the depth map in pixels. */
  width: number
  /** Height of the depth map in pixels. */
  height: number
  /** Robust lower bound of this frame's disparity values (2nd percentile). */
  low: number
  /** Robust upper bound of this frame's disparity values (98th percentile). */
  high: number
  /** Inference wall time in milliseconds (for the HUD). */
  inferenceTimeMs: number
  /** Preprocessing (crop/scale) part of {@linkcode inferenceTimeMs}. */
  prepTimeMs: number
  /** Pure CoreML prediction part of {@linkcode inferenceTimeMs}. */
  predictTimeMs: number
  /** The raw Float32 disparity values. */
  data: ArrayBuffer
}
