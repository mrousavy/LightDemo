/**
 * Live pipeline status, written by the frame-processor worklet and polled
 * by the React (main JS) runtime for the HUD.
 */
export interface LightStatus {
  frameCount: number
  fps: number
  renderTimeMs: number
  depthTimeMs: number
  handTimeMs: number
  frameWidth: number
  frameHeight: number
  frameOrientation: string
  frameMirrored: boolean
  pixelFormat: string
  lightX: number
  lightY: number
  lightZ: number
  handTracked: boolean
  pinchRatio: number
  grabbed: boolean
  depthSeq: number
  handSeq: number
}
