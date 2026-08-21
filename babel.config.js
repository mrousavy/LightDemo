// Plugin order matters: unplugin-typegpu compiles 'use gpu' kernels first,
// then the worklets transform. bundleMode + importForwarding are required for
// TypeGPU resources to transfer into worklet runtimes (see the typegpu skill,
// references/react.md): shader definitions do not transfer - worklets
// re-import them, so every module with module-scope 'use gpu' definitions
// must be listed under importForwarding.
const workletsPluginOptions = {
  bundleMode: true,
  importForwarding: {
    moduleNames: ['typegpu'],
    relativePaths: ['LightDemo/src/depthart', 'LightDemo/src'],
  },
};

module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [
    // typegpu's barrel file uses `export * as ns` which the RN preset does
    // not transform early enough.
    '@babel/plugin-transform-export-namespace-from',
    '@babel/plugin-transform-class-static-block',
    'unplugin-typegpu/babel',
    ['react-native-worklets/plugin', workletsPluginOptions],
  ],
};
