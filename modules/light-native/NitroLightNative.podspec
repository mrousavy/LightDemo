Pod::Spec.new do |s|
  s.name         = "NitroLightNative"
  s.version      = "0.1.0"
  s.summary      = "Local Nitro module for LightDemo (CoreML depth + Vision hands)"
  s.homepage     = "https://github.com/mrousavy"
  s.license      = "MIT"
  s.authors      = "Marc Rousavy"
  s.platforms    = { :ios => "15.1" }
  s.source       = { :git => "https://github.com/mrousavy/LightDemo.git", :tag => "#{s.version}" }

  s.source_files = ["ios/**/*.{swift,h,m,mm}"]

  # The .mlpackage goes through the resource-bundle target's Xcode build
  # rules, which compile it into a .mlmodelc inside the bundle.
  s.resource_bundles = {
    "LightNativeModels" => ["ios/DepthAnythingV2Small.mlpackage"]
  }

  s.frameworks = ["CoreML", "Vision", "Accelerate", "CoreVideo", "ImageIO", "UniformTypeIdentifiers"]

  load 'nitrogen/generated/ios/NitroLightNative+autolinking.rb'
  add_nitrogen_files(s)
end
