// Hermes provides `performance` at runtime (also inside worklet runtimes),
// but the narrowed tsconfig "types" list drops its declaration.
declare const performance: { now(): number }
