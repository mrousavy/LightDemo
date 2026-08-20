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
const TEMPORAL_ALPHA = 0.32;
// Original demo uses 0.8; raised so fast-moving objects leave less of a
// residual lighting trail (20% -> 8% of the previous depth under motion).
const MOTION_ALPHA = 0.92;
const MOTION_LOW = 0.02;
const MOTION_HIGH = 0.09;

const GRADIENT_RADIUS = 7;
const GRADIENT_LIMIT = 0.009;
const GRADIENT_NOISE_ENERGY = 0.0003 * 0.0003;
const OCCLUSION_TAPS = 16.0;
const OCCLUSION_SCALE = 0.07;
const OCCLUSION_RANGE = 0.25;
const OCCLUSION_FLOOR = 0.012;

const NEAR_Z = 0.0;
const SURFACE_FAR_Z = -0.7;
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
const BULB_CAMERA_Z = 2.0;
const BULB_REFERENCE_Z = 0.42;
const BULB_CORE = 8.0;
const BULB_LIMB = 0.28;
const BULB_EDGE = 0.75;
const BULB_EDGE_FLOOR = 0.004;
const BULB_EDGE_LIMIT = 0.3;
const BULB_HALO = 1.6;
const BULB_HALO_SPAN = 1.2;
const BULB_VEIL = 0.12;
const BULB_VEIL_SPAN = 4.0;
const BULB_ONSET = 0.6;
const BULB_OCCLUSION_SOFTNESS = 0.02;
const BULB_SOURCE_SOFTNESS = 0.08;
const BULB_SAMPLE_SPREAD = 0.6;

const SHADOW_FAR_Z = -1.25;
const SHADOW_STEPS = 32;
const SHADOW_SPAN = 0.3;
const SHADOW_BASELINE = 0.005;
const SHADOW_BIAS = 0.014;
const SHADOW_SLOPE_BIAS = 0.02;
const SHADOW_THICKNESS = 0.7;
const SHADOW_THICKNESS_GROWTH = 2.6;
const SHADOW_SOFTNESS = 0.089;
const SHADOW_GAIN = 2.5;
const SHADOW_FRONT_FADE = 0.2;
`;

// One shared uniform for both compute passes. 32 bytes.
const COMPUTE_PARAMS = /* wgsl */ `
struct ComputeParams {
  size: vec2u,     // depth map size in texels
  reset: u32,      // 1 = skip EMAs (first frame after camera change)
  mirror: u32,     // 1 = surface texture written mirrored in x
  range: vec2f,    // stabilized (low, high) disparity range
  _pad: vec2f,
}
`;

// Pass A: normalize disparity by the stabilized range + motion-adaptive
// temporal EMA into the persistent history buffer. The disparity texture is
// CoreML's output IOSurface imported directly into WebGPU (r16float) -
// the depth map never touches the CPU.
export const DEPTH_PREPARE_SHADER = /* wgsl */ `
${CONSTANTS}
${COMPUTE_PARAMS}

@group(0) @binding(0) var<uniform> params: ComputeParams;
@group(0) @binding(1) var disparityTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> history: array<f32>;

@compute @workgroup_size(64)
fn depthPrepare(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= params.size.x * params.size.y) { return; }
  let coord = vec2i(i32(index % params.size.x), i32(index / params.size.x));
  let low = params.range.x;
  let span = max(params.range.y - low, 0.001);
  let disp = textureLoad(disparityTex, coord, 0).r;
  var normalized = 0.0;
  if (disp == disp) { normalized = saturate((disp - low) / span); }
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
  let center = depthTexelAt(coord, size);
  let left   = depthTexelAt(coord + vec2i(-GRADIENT_RADIUS, 0), size);
  let right  = depthTexelAt(coord + vec2i( GRADIENT_RADIUS, 0), size);
  let up     = depthTexelAt(coord + vec2i(0, -GRADIENT_RADIUS), size);
  let down   = depthTexelAt(coord + vec2i(0,  GRADIENT_RADIUS), size);
  var gradient = surfaceSlope(vec2f(
    gentlerDelta(center - left, right - center),
    gentlerDelta(center - up,  down  - center)) / f32(GRADIENT_RADIUS));

  var occlusion = 0.0;
  for (var r = 0; r < 2; r++) {
    let radius = select(9, 3, r == 0);
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
  _pad0: f32,
  cropScale: vec2f,   // display uv -> camera-buffer uv (center crop)
  cropOffset: vec2f,
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
fn shadowZ(depthValue: f32) -> f32 { return mix(SHADOW_FAR_Z, NEAR_Z, depthValue); }
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

fn shadowFactor(origin: vec3f, lightDirection: vec3f, reach: f32, jitter: f32) -> f32 {
  let stride = reach / f32(SHADOW_STEPS);
  let baselineTravel = reach * (SHADOW_BASELINE / SHADOW_SPAN);
  let trailProbe = origin - lightDirection * baselineTravel;
  let receiverRise = max(
    origin.z - shadowZ(depthAt(trailProbe.xy + vec2f(0.5))) - baselineTravel * lightDirection.z,
    0.0);
  let risePerTravel = receiverRise / baselineTravel;
  var occlusion = 0.0;
  for (var step = 0; step < SHADOW_STEPS; step++) {
    let travel = (f32(step) + jitter) * stride;
    let probe = origin + lightDirection * travel;
    let sampleZ = shadowZ(depthAt(probe.xy + vec2f(0.5)));
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

fn bulbRadius() -> f32 {
  return BULB_WORLD_RADIUS * ((BULB_CAMERA_Z - BULB_REFERENCE_Z) / (BULB_CAMERA_Z - params.lightZ));
}
fn bulbExposure(radius: f32) -> f32 {
  var open = 0.0;
  for (var sy = -1; sy <= 1; sy++) {
    for (var sx = -1; sx <= 1; sx++) {
      let probe = params.lightPosition + vec2f(f32(sx), f32(sy)) * (radius * BULB_SAMPLE_SPREAD);
      open += smoothstep(0.0, BULB_SOURCE_SOFTNESS, params.lightZ - surfaceZ(depthAt(probe)));
    }
  }
  return open / 9.0;
}
fn bulbSurface(uv: vec2f, tint: vec3f, depthValue: f32) -> vec4f {
  let radius = bulbRadius();
  let spread = length(uv - params.lightPosition) / radius;
  let limb = saturate(spread);
  let dome = sqrt(max(1.0 - limb * limb, 0.0));
  let facing = dome * dome;
  let front = params.lightZ + BULB_WORLD_RADIUS * dome;
  let solid = smoothstep(0.0, BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depthValue));
  let edge = clamp(fwidth(spread) * BULB_EDGE, BULB_EDGE_FLOOR, BULB_EDGE_LIMIT);
  let coverage = (1.0 - smoothstep(1.0 - edge, 1.0 + edge, spread)) * solid;
  let hue = mix(tint, vec3f(1.0), facing * facing);
  return vec4f(hue * (BULB_CORE * mix(BULB_LIMB, 1.0, facing)), coverage);
}
fn bulbGlow(uv: vec2f, tint: vec3f) -> vec3f {
  let radius = bulbRadius();
  let radii = length(uv - params.lightPosition) / radius;
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

  let centered = uv - vec2f(0.5);
  let noise = ignDither(uv);
  let position = vec3f(centered, surfaceZ(surface.w));
  let lightPosition = vec3f(params.lightPosition - vec2f(0.5), params.lightZ);
  let toLight = lightPosition - position;
  let dist = max(length(toLight), 0.0001);
  let lightDirection = toLight / dist;
  let spread = dist / LIGHT_RADIUS;
  let falloff = 1.0 / (1.0 + spread * spread);
  let wrapped = saturate((dot(normal, lightDirection) + LIGHT_WRAP) / (1.0 + LIGHT_WRAP));
  let lambert = wrapped * wrapped;

  var shadow = 1.0;
  if (params.shadowAmount > 0.0) {
    let shadowOrigin = vec3f(centered, shadowZ(surface.w));
    let shadowToLight = lightPosition - shadowOrigin;
    let shadowDistance = max(length(shadowToLight), 0.0001);
    let reach = shadowDistance * (SHADOW_SPAN / max(length(shadowToLight.xy), SHADOW_SPAN));
    let traced = shadowFactor(shadowOrigin, shadowToLight / shadowDistance, reach, noise);
    shadow = mix(1.0, traced, params.shadowAmount);
  }
  let occlusion = mix(1.0, surface.z, params.occlusionAmount);

  let albedo = pow(cameraColor, vec3f(GAMMA));
  let tint = params.lightColor.rgb;
  let halfDirection = normalize(lightDirection + vec3f(0.0, 0.0, 1.0));
  let lobe = pow(saturate(dot(normal, halfDirection)), SPECULAR_POWER);
  let grazing = pow(1.0 - saturate(normal.z), 5.0);
  let highlight = lobe * (SPECULAR_F0 + (1.0 - SPECULAR_F0) * grazing);

  var lit = albedo * AMBIENT_FILL * (params.exposure * occlusion);
  lit += albedo * tint * (lambert * falloff * shadow * params.intensity);
  lit += tint * (highlight * falloff * shadow * occlusion * params.specularAmount * params.intensity);
  let presence = bulbPresence();
  let bulb = bulbSurface(uv, tint, surface.w);
  lit = mix(lit, bulb.rgb * presence, bulb.a * presence);
  lit += bulbGlow(uv, tint) * presence;
  let display = pow(tonemap(lit), vec3f(1.0 / GAMMA));
  return vec4f(display + vec3f((noise - 0.5) * DITHER_STEP), 1.0);
}
`;
