/**
 * One detected hand. Coordinates are normalized [0,1], origin TOP-LEFT of
 * the center-cropped region shared with the depth map.
 */
export interface TrackedHand {
  /** Whether this slot contains a detection. */
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
   * Distance between thumb tip and index tip, normalized by the hand size.
   * ~0.1-0.25 = pinching, > 0.4 = open.
   */
  pinchRatio: number
  /**
   * Wrist to middle-finger-MCP distance in normalized crop units - grows
   * ~1/distance-from-camera (the only absolute-ish depth cue).
   */
  handSize: number
  /** Minimum Vision confidence across the used joints, 0-1. */
  confidence: number
  /**
   * Raw model disparity of the hand region: an 85th-percentile probe over
   * ALL confident landmarks (fingers, knuckles, palm, wrist). Far more
   * robust than fingertip taps - a monocular model can smear thin fingers
   * into a busy background, but not the whole hand. -1 when unavailable.
   */
  disparity: number
}
