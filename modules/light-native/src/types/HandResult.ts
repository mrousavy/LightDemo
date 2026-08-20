/**
 * One completed hand-pose detection.
 *
 * All coordinates are normalized [0,1] with origin at the TOP-LEFT of the
 * center-cropped region matching the depth model's aspect ratio (so they can
 * be compared 1:1 with depth-map / canvas UVs before mirroring). Values can
 * fall slightly outside [0,1] when the hand is outside the cropped region.
 */
export interface HandResult {
  /** Monotonically increasing sequence number. -1 if no detection completed yet. */
  seq: number
  /** Whether a hand was found in this frame. */
  tracked: boolean
  /** Thumb tip. */
  thumbX: number
  thumbY: number
  /** Index finger tip. */
  indexX: number
  indexY: number
  /** Midpoint between thumb tip and index tip (the "pinch point"). */
  midX: number
  midY: number
  /**
   * Distance between thumb tip and index tip, normalized by the hand size
   * (wrist to middle-finger-MCP distance). ~0.1-0.25 = pinching, > 0.4 = open.
   */
  pinchRatio: number
  /**
   * Wrist to middle-finger-MCP distance in normalized crop units. Grows
   * proportionally to 1/distance-from-camera, making it the only
   * absolute-ish depth cue available (relative depth models normalize the
   * nearest object - usually the hand itself - to the same value).
   */
  handSize: number
  /** Minimum Vision confidence across the used joints, 0-1. */
  confidence: number
  /** Detection wall time in milliseconds (for the HUD). */
  detectionTimeMs: number
}
