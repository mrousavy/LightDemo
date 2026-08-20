import { NitroModules } from 'react-native-nitro-modules'
import type { LightNative } from './specs/LightNative.nitro'

export const LightNativeModule =
  NitroModules.createHybridObject<LightNative>('LightNative')

export type { LightNative } from './specs/LightNative.nitro'
export type { LightPipeline } from './specs/LightPipeline.nitro'
export type { DepthResult } from './types/DepthResult'
export type { HandResult } from './types/HandResult'
export type { TrackedHand } from './types/TrackedHand'
export type { LightControls } from './types/LightControls'
export type { LightStatus } from './types/LightStatus'
