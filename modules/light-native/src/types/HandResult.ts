import type { TrackedHand } from './TrackedHand'

/**
 * One completed hand-pose detection pass (up to two hands).
 * Slot order is NOT stable across frames - Vision assigns no identities;
 * consumers must match hands by proximity if continuity matters.
 */
export interface HandResult {
  /** Monotonically increasing sequence number. -1 if no detection completed yet. */
  seq: number
  /** First detected hand (tracked=false when absent). */
  hand1: TrackedHand
  /** Second detected hand (tracked=false when absent). */
  hand2: TrackedHand
  /** Detection wall time in milliseconds (for the HUD). */
  detectionTimeMs: number
}
