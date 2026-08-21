// WGSL port of the TypeGPU "monocular light injection" demo
// (apps/typegpu-docs/src/examples/image-processing/monocular-light-injection),
// adapted for react-native-webgpu + VisionCamera:
// - camera arrives as a texture_external (NV12 IOSurface, YUV->RGB by Dawn)
// - disparity arrives as a tightly packed array<f32> storage buffer
//   (uploaded from the CoreML Depth Anything V2 output)
// - the 2%..98% percentile range is computed on the CPU (Swift) and the
//   range EMA in JS; the per-pixel motion-adaptive EMA stays on the GPU
// - mirroring (front camera) is applied consistently: the surface pass
//   writes the surface texture already display-oriented, and the relight
//   pass mirrors the camera fetch with the same flag.

const CONSTANTS = /* wgsl */ `
// The lighting field (history + surface texture) runs at FIELD_SCALE x the
// depth model resolution with bilinear depth upsampling: normals, AO and
// shadows computed at raw model resolution look blocky against the sharp
// camera image. Gradient/AO step radii stay in MODEL texel units.
const FIELD_SCALE = 3;
const TEMPORAL_ALPHA = 0.32;
// Original demo uses 0.8; raised so fast-moving objects leave less of a
// residual lighting trail (20% -> 8% of the previous depth under motion).
const MOTION_ALPHA = 0.92;
const MOTION_LOW = 0.02;
const MOTION_HIGH = 0.09;

// Joint bilateral upsampling (depthPrepare): gaussian sharpness terms.
// JBU_SPATIAL = 1/(2*sigma^2) with sigma ~0.9 model texels; JBU_RANGE with
// sigma ~0.09 in luma; JBU_FLOOR keeps a bilinear fallback where the camera
// offers no contrast so flat regions degrade to a plain gaussian upsample.
const JBU_SPATIAL = 0.62;
const JBU_RANGE = 60.0;
const JBU_FLOOR = 0.02;

const GRADIENT_RADIUS = 7;
const GRADIENT_LIMIT = 0.009;
const GRADIENT_NOISE_ENERGY = 0.0003 * 0.0003;
const OCCLUSION_TAPS = 16.0;
const OCCLUSION_SCALE = 0.07;
const OCCLUSION_RANGE = 0.25;
const OCCLUSION_FLOOR = 0.012;

// ONE unified z-space for EVERYTHING (shading, shadows, occlusion, bulb,
// and the JS light control): normalized disparity 1 (nearest object) maps
// to z = 0, disparity 0 (farthest) to z = SURFACE_FAR_Z. Positive z is
// toward the viewer, in front of the whole scene. The original demo used a
// second, exaggerated space just for shadows (far -1.25 vs -0.7), which
// forced error-prone conversions of the light's z between spaces; with one
// space the light is trivially "inside" the reconstructed scene.
// The magnitude also sets relief realism: with a typical desk scene
// spanning ~2m of depth and the image height ~0.8m at face distance,
// depth range / image height is ~1.1 - which is exactly this constant
// (world x/y units are image-height = 1).
const NEAR_Z = 0.0;
const SURFACE_FAR_Z = -1.1;
const LIGHT_RADIUS = 0.85;
const LIGHT_WRAP = 0.25;
const RELIEF_SCALE = 200.0;
const SLOPE_COMPRESSION = 0.55;
const SPECULAR_POWER = 36.0;
const SPECULAR_F0 = 0.06;
const GAMMA = 2.2;
const WHITE_POINT = 2.6;
const LUMINANCE_WEIGHTS = vec3f(0.2126, 0.7152, 0.0722);
const HIGHLIGHT_BLEACH = 2.0;
const AMBIENT_FILL = vec3f(0.78, 0.86, 1.0);
const DITHER_STEP = 1.0 / 255.0;

const BULB_WORLD_RADIUS = 0.05;
const BULB_CORE = 5.0;
const BULB_LIMB = 0.12;
const BULB_EDGE = 0.75;
const BULB_EDGE_FLOOR = 0.004;
const BULB_EDGE_LIMIT = 0.3;
const BULB_HALO = 1.6;
const BULB_HALO_SPAN = 1.2;
const BULB_VEIL = 0.12;
const BULB_VEIL_SPAN = 4.0;
const BULB_ONSET = 0.6;
const BULB_TURB_FREQ = 3.0;
const BULB_TURB_SPEED = 0.35;
const BULB_TURB_AMOUNT = 0.22;
const BULB_OCCLUSION_SOFTNESS = 0.02;
const BULB_SOURCE_SOFTNESS = 0.08;
const BULB_SAMPLE_SPREAD = 0.6;

const SHADOW_STEPS = 40;
const SHADOW_SPAN = 0.3;
const SHADOW_BASELINE = 0.005;
const SHADOW_BIAS = 0.022;
const SHADOW_SLOPE_BIAS = 0.032;
const SHADOW_THICKNESS = 0.7;
const SHADOW_THICKNESS_GROWTH = 2.6;
const SHADOW_SOFTNESS = 0.15;
const SHADOW_GAIN = 2.1;
const SHADOW_FRONT_FADE = 0.2;
`;

// One shared uniform for both compute passes. 48 bytes.
const COMPUTE_PARAMS = /* wgsl */ `
struct ComputeParams {
  size: vec2u,        // FIELD size in texels
  reset: u32,         // 1 = skip EMAs (first frame after camera change)
  mirror: u32,        // 1 = surface texture written mirrored in x
  modelSize: vec2f,   // disparity grid size in texels (anamorphic full-frame)
  cropScale: vec2f,   // model uv -> oriented camera uv (identity: full frame)
  cropOffset: vec2f,
  _pad: vec2f,
}
`;

// Pass A: normalize disparity by the GPU-computed 2-98% range +
// motion-adaptive temporal EMA into the persistent history buffer. The
// disparity comes straight from the DepthART inference output buffer (hwc4:
// one vec4 per pixel, value in .x) and the range from the GPU histogram
// estimator - the depth map never exists anywhere but GPU memory.
export const DEPTH_PREPARE_SHADER = /* wgsl */ `
${CONSTANTS}
${COMPUTE_PARAMS}

@group(0) @binding(0) var<uniform> params: ComputeParams;
@group(0) @binding(1) var<storage, read> disparity: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> history: array<f32>;
@group(0) @binding(3) var disparitySampler: sampler;
@group(0) @binding(4) var cameraTex: texture_external;
@group(0) @binding(5) var<storage, read> disparityRange: vec2f;
// Model-grid luma written by the depth preprocess pass - the JBU guide at
// tap positions, WITHOUT 16 external-texture (YUV-convert) samples/texel.
@group(0) @binding(6) var<storage, read> modelLuma: array<f32>;

fn disparityAt(coord: vec2f) -> f32 {
  let c = vec2i(clamp(coord, vec2f(0.0), params.modelSize - 1.0));
  return disparity[u32(c.y) * u32(params.modelSize.x) + u32(c.x)].x;
}

fn modelLumaAt(coord: vec2f) -> f32 {
  let c = vec2i(clamp(coord, vec2f(0.0), params.modelSize - 1.0));
  return modelLuma[u32(c.y) * u32(params.modelSize.x) + u32(c.x)];
}

fn cameraLuma(uv: vec2f) -> f32 {
  let c = textureSampleBaseClampToEdge(
    cameraTex, disparitySampler, params.cropOffset + uv * params.cropScale).rgb;
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(64)
fn depthPrepare(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= params.size.x * params.size.y) { return; }
  let coord = vec2i(i32(index % params.size.x), i32(index / params.size.x));
  let low = disparityRange.x;
  let span = max(disparityRange.y - low, 0.001);
  let fieldUv = (vec2f(coord) + 0.5) / vec2f(params.size);
  // Joint bilateral upsampling: reconstruct the model-resolution disparity
  // at field resolution GUIDED BY THE CAMERA IMAGE. Plain bilinear smears
  // depth edges across multi-pixel ramps that the AO/shadow terms amplify
  // into blocky fringes at silhouettes (arms, head). Weighting each model
  // texel by how much its camera pixel resembles ours snaps depth edges to
  // IMAGE edges - sub-model-texel silhouettes from the same model output.
  let mp = fieldUv * params.modelSize - 0.5;
  let baseTexel = floor(mp);
  let centerLuma = cameraLuma(fieldUv);
  var accum = 0.0;
  var weightSum = 0.0;
  for (var dy = -1; dy <= 2; dy++) {
    for (var dx = -1; dx <= 2; dx++) {
      let tap = baseTexel + vec2f(f32(dx), f32(dy));
      let disp = disparityAt(tap);
      let toTap = mp - tap;
      let spatial = exp(-dot(toTap, toTap) * JBU_SPATIAL);
      let lumaDelta = modelLumaAt(tap) - centerLuma;
      let similarity = exp(-lumaDelta * lumaDelta * JBU_RANGE) + JBU_FLOOR;
      let w = spatial * similarity;
      accum += disp * w;
      weightSum += w;
    }
  }
  var normalized = 0.0;
  if (weightSum > 0.0) {
    normalized = saturate((accum / weightSum - low) / span);
  }
  var filtered = normalized;
  if (params.reset == 0u) {
    let previous = history[index];
    let motion = smoothstep(MOTION_LOW, MOTION_HIGH, abs(normalized - previous));
    filtered = mix(previous, normalized, mix(TEMPORAL_ALPHA, MOTION_ALPHA, motion));
  }
  history[index] = filtered;
}
`;

// Pass B: depth gradient + height-field occlusion -> rgba16float surface
// texture (gradX, gradY, 1-AO, depth), written in DISPLAY orientation
// (mirror applied here so every later pass can use plain display UVs).
export const SURFACE_SHADER = /* wgsl */ `
${CONSTANTS}
${COMPUTE_PARAMS}

@group(0) @binding(0) var<uniform> params: ComputeParams;
@group(0) @binding(1) var<storage, read> depth: array<f32>;
@group(0) @binding(2) var surface: texture_storage_2d<rgba16float, write>;

fn texelIndex(coord: vec2i, size: vec2i) -> u32 {
  let c = clamp(coord, vec2i(0), size - 1);
  return u32(c.y) * u32(size.x) + u32(c.x);
}
fn depthTexelAt(coord: vec2i, size: vec2i) -> f32 {
  return depth[texelIndex(coord, size)];
}
fn gentlerDelta(backward: f32, forward: f32) -> f32 {
  let back = abs(backward);
  let front = abs(forward);
  return (backward * front + forward * back) / max(back + front, 1e-9);
}
fn surfaceSlope(gradient: vec2f) -> vec2f {
  let steepness = max(length(gradient), 1e-9);
  let shrunk = sqrt(max(steepness * steepness - GRADIENT_NOISE_ENERGY, 0.0));
  let ceiling = GRADIENT_LIMIT * tanh(shrunk / GRADIENT_LIMIT);
  return gradient * (ceiling / steepness);
}

@compute @workgroup_size(8, 8)
fn surfacePass(@builtin(global_invocation_id) gid: vec3u) {
  let size = vec2i(params.size);
  let coord = vec2i(gid.xy);
  if (coord.x >= size.x || coord.y >= size.y) { return; }
  // Step radii are in MODEL texel units (the constants were tuned for the
  // ~448px demo field), so multiply by FIELD_SCALE for field texels but
  // normalize the gradient by the model-texel distance.
  let stepRadius = GRADIENT_RADIUS * FIELD_SCALE;
  let center = depthTexelAt(coord, size);
  let left   = depthTexelAt(coord + vec2i(-stepRadius, 0), size);
  let right  = depthTexelAt(coord + vec2i( stepRadius, 0), size);
  let up     = depthTexelAt(coord + vec2i(0, -stepRadius), size);
  let down   = depthTexelAt(coord + vec2i(0,  stepRadius), size);
  var gradient = surfaceSlope(vec2f(
    gentlerDelta(center - left, right - center),
    gentlerDelta(center - up,  down  - center)) / f32(GRADIENT_RADIUS));

  var occlusion = 0.0;
  for (var r = 0; r < 2; r++) {
    let radius = select(9, 3, r == 0) * FIELD_SCALE;
    for (var sy = -1; sy <= 1; sy++) {
      for (var sx = -1; sx <= 1; sx++) {
        if (sx != 0 || sy != 0) {
          let neighbor = depthTexelAt(coord + vec2i(sx * radius, sy * radius), size);
          let difference = neighbor - center;
          let contact = 1.0 - saturate(abs(difference) / OCCLUSION_RANGE);
          let cleared = max(difference - OCCLUSION_FLOOR, 0.0);
          occlusion += saturate(cleared / OCCLUSION_SCALE) * contact;
        }
      }
    }
  }

  var outCoord = gid.xy;
  if (params.mirror != 0u) {
    outCoord.x = u32(size.x) - 1u - gid.x;
    gradient.x = -gradient.x;
  }
  textureStore(surface, outCoord,
    vec4f(gradient, 1.0 - saturate(occlusion / OCCLUSION_TAPS), center));
}
`;

// Pass C: the relight fragment (+ fullscreen triangle vertex).
// uv: [0,1]^2, origin TOP-LEFT, y down - identical to the TypeGPU demo, so
// light placement math carries over 1:1.
export const RELIGHT_SHADER = /* wgsl */ `
${CONSTANTS}

struct RelightParams {
  lightColor: vec4f,
  lightPosition: vec2f,
  lightZ: f32,
  exposure: f32,
  intensity: f32,
  relief: f32,
  specularAmount: f32,
  shadowAmount: f32,
  occlusionAmount: f32,
  mode: u32,          // 0 relit, 1 camera, 2 depth, 3 normals
  mirror: u32,
  aspect: f32,        // canvas width / height
  cropScale: vec2f,   // display uv -> camera-buffer uv (center crop)
  cropOffset: vec2f,
  time: f32,          // seconds, for the bulb flicker
  bulbScale: f32,     // on-screen size factor (hand-tracked perspective)
  _pad2: vec2f,
}

// All distance math runs in aspect-corrected "world" units (y-height = 1):
// the original demo forces a SQUARE canvas so plain UV distances work; on a
// non-square canvas UV circles render as ellipses (the squished bulb) and
// light falloff turns anisotropic.
fn worldFromUv(uv: vec2f) -> vec2f {
  return vec2f((uv.x - 0.5) * params.aspect, uv.y - 0.5);
}
fn uvFromWorld(w: vec2f) -> vec2f {
  return vec2f(w.x / params.aspect + 0.5, w.y + 0.5);
}

// Gentle filament flicker: a few incommensurate sines, +-3%.
fn flicker(t: f32) -> f32 {
  return 1.0 + 0.022 * sin(t * 7.3) * sin(t * 12.7 + 1.7)
             + 0.014 * sin(t * 23.0 + 0.5);
}

// Cheap value noise for the bulb's internal shimmer.
fn hash21(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}
fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash21(i), hash21(i + vec2f(1.0, 0.0)), u.x),
    mix(hash21(i + vec2f(0.0, 1.0)), hash21(i + vec2f(1.0, 1.0)), u.x),
    u.y);
}

@group(0) @binding(0) var<uniform> params: RelightParams;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var surfaceTex: texture_2d<f32>;
@group(0) @binding(3) var frame: texture_external;

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  var uvs = array<vec2f, 3>(
    vec2f(0.0, 2.0), vec2f(0.0, 0.0), vec2f(2.0, 0.0));
  var out: VsOut;
  out.position = vec4f(positions[vid], 0.0, 1.0);
  out.uv = uvs[vid];
  return out;
}

fn surfaceZ(depthValue: f32) -> f32 { return mix(SURFACE_FAR_Z, NEAR_Z, depthValue); }
fn depthAt(uv: vec2f) -> f32 { return textureSampleLevel(surfaceTex, samp, uv, 0.0).w; }

// display uv -> camera-buffer uv: mirror, then center-crop to model aspect.
fn cameraUvAt(uv: vec2f) -> vec2f {
  var framed = uv;
  if (params.mirror != 0u) { framed.x = 1.0 - framed.x; }
  return params.cropOffset + framed * params.cropScale;
}

fn ignDither(uv: vec2f) -> f32 {
  let p = uv * 1024.0;
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// origin/lightDirection are in aspect-corrected world space; texture
// lookups convert back through uvFromWorld. The march runs in the SAME
// unified z-space as the shading, so the light's z needs no conversion.
fn shadowFactor(origin: vec3f, lightDirection: vec3f, reach: f32, jitter: f32) -> f32 {
  let stride = reach / f32(SHADOW_STEPS);
  let baselineTravel = reach * (SHADOW_BASELINE / SHADOW_SPAN);
  let trailProbe = origin - lightDirection * baselineTravel;
  let receiverRise = max(
    origin.z - surfaceZ(depthAt(uvFromWorld(trailProbe.xy))) - baselineTravel * lightDirection.z,
    0.0);
  let risePerTravel = receiverRise / baselineTravel;
  var occlusion = 0.0;
  for (var step = 0; step < SHADOW_STEPS; step++) {
    let travel = (f32(step) + jitter) * stride;
    let probe = origin + lightDirection * travel;
    let sampleZ = surfaceZ(depthAt(uvFromWorld(probe.xy)));
    let difference = sampleZ - probe.z;
    let bias = SHADOW_BIAS + travel * (SHADOW_SLOPE_BIAS + risePerTravel);
    let thickness = SHADOW_THICKNESS * (1.0 + (travel / SHADOW_SPAN) * SHADOW_THICKNESS_GROWTH);
    if (difference > bias && difference < thickness) {
      let behindLight = 1.0 - saturate((sampleZ - params.lightZ) / SHADOW_FRONT_FADE);
      occlusion += saturate((difference - bias) / SHADOW_SOFTNESS) * behindLight;
    }
  }
  return 1.0 - saturate((occlusion / f32(SHADOW_STEPS)) * SHADOW_GAIN);
}

fn depthRamp(value: f32) -> vec3f {
  let cold = vec3f(0.03, 0.02, 0.12);
  let middle = vec3f(0.11, 0.45, 0.94);
  let warm = vec3f(0.85, 0.36, 0.96);
  let hot = vec3f(0.97, 0.97, 0.87);
  if (value < 0.4) { return mix(cold, middle, value / 0.4); }
  if (value < 0.75) { return mix(middle, warm, (value - 0.4) / 0.35); }
  return mix(warm, hot, (value - 0.75) / 0.25);
}

// Screen-space bulb radius. The scale factor comes from JS: while a hand
// holds/steers the light it is the HAND's angular size (proportional to
// 1/distance-from-camera - the bulb grows by exactly the factor the hand
// does), otherwise a near-camera perspective on the light's z.
fn bulbRadius() -> f32 {
  return BULB_WORLD_RADIUS * params.bulbScale;
}
fn bulbExposure(radius: f32) -> f32 {
  let lightWorld = worldFromUv(params.lightPosition);
  var open = 0.0;
  for (var sy = -1; sy <= 1; sy++) {
    for (var sx = -1; sx <= 1; sx++) {
      let probe = lightWorld + vec2f(f32(sx), f32(sy)) * (radius * BULB_SAMPLE_SPREAD);
      open += smoothstep(0.0, BULB_SOURCE_SOFTNESS, params.lightZ - surfaceZ(depthAt(uvFromWorld(probe))));
    }
  }
  return open / 9.0;
}
fn bulbSurface(uv: vec2f, tint: vec3f, depthValue: f32) -> vec4f {
  let radius = bulbRadius();
  let offset = (worldFromUv(uv) - worldFromUv(params.lightPosition)) / radius;
  let spread = length(offset);
  let limb = saturate(spread);
  let dome = sqrt(max(1.0 - limb * limb, 0.0));
  let front = params.lightZ + BULB_WORLD_RADIUS * dome;
  let solid = smoothstep(0.0, BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depthValue));
  let edge = clamp(fwidth(spread) * BULB_EDGE, BULB_EDGE_FLOOR, BULB_EDGE_LIMIT);
  let coverage = (1.0 - smoothstep(1.0 - edge, 1.0 + edge, spread)) * solid;

  // Blackbody-style radial temperature: white-hot core cooling through the
  // tint into a deep ember at the limb - a frosted incandescent globe
  // instead of a flat white disc.
  let core = vec3f(1.0, 0.97, 0.9);
  let ember = tint * vec3f(0.95, 0.55, 0.3);
  var glass = mix(core, tint, smoothstep(0.05, 0.6, spread));
  glass = mix(glass, ember, smoothstep(0.55, 0.95, spread));

  // Slow convective shimmer inside the globe: two octaves of value noise
  // drifting upward, masked to the mid-radius band so the core stays clean
  // and the rim stays smooth.
  let drift = params.time * BULB_TURB_SPEED;
  let swirl = offset + vec2f(drift * 0.6, -drift);
  let turb = vnoise(swirl * BULB_TURB_FREQ) * 0.67 +
             vnoise(swirl * (BULB_TURB_FREQ * 2.3) + vec2f(17.0)) * 0.33;
  let band = smoothstep(0.12, 0.45, spread) * (1.0 - smoothstep(0.75, 1.0, spread));
  let shimmer = 1.0 + (turb - 0.5) * 2.0 * BULB_TURB_AMOUNT * band;

  // Limb darkening steeper than physical (dome^2): the tonemapper clips
  // the interior to white anyway, so an early falloff is what actually
  // keeps the amber limb and the shimmer visible on screen.
  let brightness = BULB_CORE * mix(BULB_LIMB, 1.0, dome * dome) * shimmer;
  return vec4f(glass * brightness, coverage);
}
fn bulbGlow(uv: vec2f, tint: vec3f) -> vec3f {
  let radius = bulbRadius();
  let radii = length(worldFromUv(uv) - worldFromUv(params.lightPosition)) / radius;
  let halo = exp(-radii / BULB_HALO_SPAN);
  let veil = exp(-radii / BULB_VEIL_SPAN);
  return tint * ((halo * BULB_HALO + veil * BULB_VEIL) * bulbExposure(radius));
}
fn bulbPresence() -> f32 { return saturate(params.intensity / BULB_ONSET); }

fn compress(value: f32) -> f32 {
  return (value * (value / (WHITE_POINT * WHITE_POINT) + 1.0)) / (value + 1.0);
}
fn tonemap(color: vec3f) -> vec3f {
  let luminance = max(dot(color, LUMINANCE_WEIGHTS), 0.0001);
  let mapped = compress(luminance);
  let shoulder = color / vec3f(WHITE_POINT * WHITE_POINT) + vec3f(1.0);
  let perChannel = (color * shoulder) / (color + vec3f(1.0));
  let bleach = pow(saturate(mapped), HIGHLIGHT_BLEACH);
  return saturate(mix(color * (mapped / luminance), perChannel, bleach));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
  let uv = in.uv;
  let cameraColor = saturate(
    textureSampleBaseClampToEdge(frame, samp, cameraUvAt(uv)).rgb);
  if (params.mode == 1u) { return vec4f(cameraColor, 1.0); }

  let surface = textureSampleLevel(surfaceTex, samp, uv, 0.0);
  if (params.mode == 2u) { return vec4f(depthRamp(saturate(surface.w)), 1.0); }

  let slope = surface.xy * (params.relief * RELIEF_SCALE);
  let tilt = -slope / (1.0 + length(slope) * SLOPE_COMPRESSION);
  let normal = normalize(vec3f(tilt, 1.0));
  if (params.mode == 3u) { return vec4f(normal * 0.5 + 0.5, 1.0); }

  let world = worldFromUv(uv);
  let noise = ignDither(uv);
  let position = vec3f(world, surfaceZ(surface.w));
  let lightPosition = vec3f(worldFromUv(params.lightPosition), params.lightZ);
  let toLight = lightPosition - position;
  let dist = max(length(toLight), 0.0001);
  let lightDirection = toLight / dist;
  let spread = dist / LIGHT_RADIUS;
  let falloff = 1.0 / (1.0 + spread * spread);
  let wrapped = saturate((dot(normal, lightDirection) + LIGHT_WRAP) / (1.0 + LIGHT_WRAP));
  let lambert = wrapped * wrapped;

  var shadow = 1.0;
  if (params.shadowAmount > 0.0) {
    // Same space as the shading: march straight from the shaded point
    // toward the actual light position.
    let reach = dist * (SHADOW_SPAN / max(length(toLight.xy), SHADOW_SPAN));
    let traced = shadowFactor(position, lightDirection, reach, noise);
    // Area-light physics: the bulb (radius 0.05) subtends a huge solid
    // angle for surfaces right next to it, so shadows fade out near the
    // light instead of blackening a face the bulb is hovering in front of.
    let nearLightFade = saturate((dist - BULB_WORLD_RADIUS * 2.5) / 0.25);
    shadow = mix(1.0, traced, params.shadowAmount * nearLightFade);
  }
  let occlusion = mix(1.0, surface.z, params.occlusionAmount);

  let albedo = pow(cameraColor, vec3f(GAMMA));
  let tint = params.lightColor.rgb;
  let halfDirection = normalize(lightDirection + vec3f(0.0, 0.0, 1.0));
  let lobe = pow(saturate(dot(normal, halfDirection)), SPECULAR_POWER);
  let grazing = pow(1.0 - saturate(normal.z), 5.0);
  let highlight = lobe * (SPECULAR_F0 + (1.0 - SPECULAR_F0) * grazing);

  // Filament flicker: light emission and the bulb surface breathe together.
  let flick = flicker(params.time);
  var lit = albedo * AMBIENT_FILL * (params.exposure * occlusion);
  lit += albedo * tint * (lambert * falloff * shadow * params.intensity * flick);
  lit += tint * (highlight * falloff * shadow * occlusion * params.specularAmount * params.intensity * flick);
  let presence = bulbPresence();
  let bulb = bulbSurface(uv, tint, surface.w);
  lit = mix(lit, bulb.rgb * presence * flick, bulb.a * presence);
  lit += bulbGlow(uv, tint) * presence * flick;
  let display = pow(tonemap(lit), vec3f(1.0 / GAMMA));
  return vec4f(display + vec3f((noise - 0.5) * DITHER_STEP), 1.0);
}
`;
