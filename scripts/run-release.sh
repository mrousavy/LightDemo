#!/bin/zsh
# Build-products launcher for the RELEASE configuration.
#
# Two hard-won gotchas live here:
# 1. Incremental Release builds BREAK the code seal: the RN "Bundle React
#    Native code and images" phase re-runs every build (no declared outputs)
#    and overwrites main.jsbundle after Xcode already signed the app. AMFI
#    then refuses the app in ways that look unrelated (dyld stall, launchd
#    error 162, Finder's "damaged" dialog). Always verify and re-sign.
# 2. Launch via `open` on a wrapper in the BUILD PRODUCTS dir. Never
#    double-click a wrapper in Finder and never launch a copy that sits in
#    ~/Downloads - Gatekeeper assesses those paths and blocklists the app.
set -e
cd "$(dirname "$0")/.."
RELAPP="$(pwd)/ios/build-release/Build/Products/Release-iphoneos/LightDemo.app"
WRAPPED="$(pwd)/ios/build/Build/Products/Debug-iphoneos/LightDemoWrapped.app"
IDENTITY="Apple Development: Marc Rousavy (2N4LAL9NBE)"

if ! codesign --verify --deep --strict "$RELAPP" 2>/dev/null; then
  echo "seal broken (RN bundle phase ran after signing) - re-signing"
  codesign -f --preserve-metadata=identifier,entitlements,flags -s "$IDENTITY" "$RELAPP"
  codesign --verify --deep --strict "$RELAPP"
fi

pkill -x LightDemo 2>/dev/null || true
sleep 1

if ! cmp -s "$RELAPP/LightDemo" "$WRAPPED/Wrapper/LightDemo.app/LightDemo" 2>/dev/null; then
  echo "recreating wrapper with release app"
  rm -rf "$WRAPPED"
  mkdir -p "$WRAPPED/Wrapper"
  cp -R "$RELAPP" "$WRAPPED/Wrapper/"
  ln -s Wrapper/LightDemo.app "$WRAPPED/WrappedBundle"
fi

open "$WRAPPED"
echo "launched (release)"
