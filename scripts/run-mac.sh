#!/bin/zsh
# Launch the freshly built LightDemo.app on this Mac (Designed for iPad).
# iOS apps need the LaunchServices "Wrapper" structure to run on macOS.
# The wrapper is only recreated when the build actually changed - recreating
# it on every launch invalidated the ANE model-compilation cache and cost
# ~10s of Neural Engine recompilation per startup.
set -e
P="$(dirname "$0")/../ios/build/Build/Products/Debug-iphoneos"
P="$(cd "$P" && pwd)"
SRC="$P/LightDemo.app"
WRAPPED="$P/LightDemoWrapped.app"
INNER="$WRAPPED/Wrapper/LightDemo.app"

pkill -x LightDemo 2>/dev/null || true
sleep 1

if ! cmp -s "$SRC/LightDemo" "$INNER/LightDemo" 2>/dev/null; then
  echo "build changed - recreating wrapper"
  rm -rf "$WRAPPED"
  mkdir -p "$WRAPPED/Wrapper"
  cp -R "$SRC" "$WRAPPED/Wrapper/"
  ln -s Wrapper/LightDemo.app "$WRAPPED/WrappedBundle"
fi

open "$WRAPPED"
echo "launched"
