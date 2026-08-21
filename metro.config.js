const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const { getBundleModeMetroConfig } = require('react-native-worklets/bundleMode');

const defaultConfig = getDefaultConfig(__dirname);

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  resolver: {
    // The DepthART weight bundle ships as a Metro asset and is fetched as an
    // ArrayBuffer at startup.
    assetExts: [...defaultConfig.resolver.assetExts, 'depthart'],
  },
};

// Worklets bundleMode (required for TypeGPU resource transfer into the frame
// worklet) needs its Metro resolver/serializer hooks.
module.exports = getBundleModeMetroConfig(mergeConfig(defaultConfig, config));
