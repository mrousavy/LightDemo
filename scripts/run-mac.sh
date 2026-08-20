#!/bin/zsh
# Launch the freshly built LightDemo.app on this Mac (Designed for iPad).
# iOS apps need the LaunchServices "Wrapper" structure to run on macOS.
set -e
P="$(dirname "$0")/../ios/build/Build/Products/Debug-iphoneos"
P="$(cd "$P" && pwd)"
pkill -x LightDemo 2>/dev/null || true
sleep 1
rm -rf "$P/LightDemoWrapped.app"
mkdir -p "$P/LightDemoWrapped.app/Wrapper"
cp -R "$P/LightDemo.app" "$P/LightDemoWrapped.app/Wrapper/"
ln -s Wrapper/LightDemo.app "$P/LightDemoWrapped.app/WrappedBundle"
open "$P/LightDemoWrapped.app"
echo "launched"
