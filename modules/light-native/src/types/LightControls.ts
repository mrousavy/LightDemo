/**
 * UI-controlled parameters, written by the React (main JS) runtime and read
 * by the frame-processor worklet every frame. Stored natively so the worklet
 * closure never needs to be re-created when the UI changes.
 */
export interface LightControls {
  /** 0 = relit, 1 = camera, 2 = depth, 3 = normals. */
  mode: number
  intensity: number
  exposure: number
  relief: number
  specular: number
  shadow: number
  occlusion: number
  /** Light color (linear RGB 0-1). */
  colorR: number
  colorG: number
  colorB: number
  /** Manual light placement (display UV). Applied while touchActive. */
  touchX: number
  touchY: number
  touchActive: boolean
  /** Manual light depth from the slider, in surface-Z space. */
  lightZ: number
  /** Enable hand tracking control. */
  handControl: boolean
  /** Take a window snapshot to this path on the next frame ("" = off). */
  snapshotPath: string
}
