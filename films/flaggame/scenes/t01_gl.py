"""t01 — the physical world around the tract: an OpenGL (moderngl / EGL, headless) photographic renderer.
Owner: T01Tract.

Grey, wet, raining, shallow depth of field.  World frame in cm: x right, y forward, z up.
Passes per frame:
  1. top-down "tract height" map of the paper (soft contact shadows + splash heights)
  2. scene (4x MSAA): backdrop billboard, mud terrain (near + far heightfields, wet GGX, puddles with
     ray-traced-looking rain ripples, reflections of an overcast sky and the camp), flag layer
     (screen-space vx.flag canvas placed at its depth), the tract leaves (page textures, wet paper,
     water beads that magnify the print, show-through, mud smears) -> colour + linear depth + flag mask
  3. depth-of-field gather (scatter-as-gather, 96 taps) on colour and flag mask
  4. rain streaks + splash crowns (depth-tested against the scene), MASKED OUT of every flag pixel
  5. tonemap -> sRGB
Flags: the only colour. Page textures carry a flag-coverage alpha: over those texels there are NO beads,
NO specular, NO mud, NO show-through, NO rain — just the plain yellow under smooth light.
A context is created lazily per process (forked render workers each get their own).
"""
import math
import os

import numpy as np

_CTX = {}

# ----------------------------------------------------------------------------------------------- math
def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, np.float64)
    f = np.asarray(target, np.float64) - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fovy_deg, aspect, near=0.8, far=6000.0):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    M = np.zeros((4, 4))
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = 2 * far * near / (near - far)
    M[3, 2] = -1
    return M


class Cam:
    """perspective camera; focus = focus distance (cm); blur = CoC scale in design px at infinity"""

    def __init__(self, eye, target, fovy=26.0, focus=None, blur=34.0, roll=0.0, up=(0.0, 0.0, 1.0)):
        self.eye = np.asarray(eye, np.float64)
        self.target = np.asarray(target, np.float64)
        self.up = np.asarray(up, np.float64)
        self.fovy = float(fovy)
        self.focus = float(focus if focus is not None else np.linalg.norm(self.target - self.eye))
        self.blur = float(blur)
        self.roll = float(roll)

    def view(self):
        V = look_at(self.eye, self.target, self.up)
        if self.roll:
            c, s = math.cos(self.roll), math.sin(self.roll)
            R = np.eye(4)
            R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
            V = R @ V
        return V

    def proj(self, aspect=16 / 9):
        return perspective(self.fovy, aspect)

    def vp(self):
        return self.proj() @ self.view()

    def project(self, p):
        """world point(s) -> design screen (x, y) (1920x1080) and view depth"""
        p = np.atleast_2d(np.asarray(p, np.float64))
        ph = np.concatenate([p, np.ones((len(p), 1))], 1)
        v = (self.view() @ ph.T).T
        c = (self.proj() @ v.T).T
        ndc = c[:, :3] / c[:, 3:4]
        sx = (ndc[:, 0] * 0.5 + 0.5) * 1920.0
        sy = (1 - (ndc[:, 1] * 0.5 + 0.5)) * 1080.0
        return np.stack([sx, sy], 1), -v[:, 2]

    def px_per_cm(self, depth):
        """design px per cm at view depth"""
        return 540.0 / math.tan(math.radians(self.fovy) / 2) / depth


# ----------------------------------------------------------------------------------------------- GLSL
COMMON = r"""
#version 330
uniform sampler2D u_hnear;
uniform sampler2D u_hfar;
uniform vec4 u_near_rect;
uniform vec4 u_far_rect;
uniform float u_water;
uniform float u_time;
uniform vec3 u_eye;
uniform vec3 u_sun;
uniform sampler2D u_backdrop;
uniform vec4 u_bd;         // backdrop plane: y, x-center, width, z-bottom
uniform float u_bd_h;      // backdrop height
uniform sampler2D u_tmap;  // top-down tract map: r = coverage, g = height above ground (cm)
uniform vec4 u_tmap_rect;
uniform float u_rain;      // rain intensity 0..1

float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * .1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973)); p3 += dot(p3, p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
float vnoise(vec2 p){ vec2 i = floor(p); vec2 f = fract(p); vec2 u = f*f*(3.0-2.0*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y); }

float nearW(vec2 p){
  vec2 un = (p - u_near_rect.xy) / u_near_rect.zw;
  vec2 e = abs(un - 0.5) * 2.0;
  return clamp((max(e.x, e.y) - 0.88) / 0.1, 0.0, 1.0);
}
float Hn(vec2 p){ return texture(u_hnear, (p - u_near_rect.xy) / u_near_rect.zw).r; }
float Hf(vec2 p){ return texture(u_hfar, (p - u_far_rect.xy) / u_far_rect.zw).r; }
float Hgt(vec2 p){ float w = nearW(p); return w >= 1.0 ? Hf(p) : mix(Hn(p), Hf(p), w); }

vec4 tmap(vec2 p, float lod){ vec2 uv = (p - u_tmap_rect.xy) / u_tmap_rect.zw;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return vec4(0.0);
  return textureLod(u_tmap, uv, lod); }

vec3 sky(vec3 d){
  float e = d.z;
  float base = mix(0.50, 1.05, smoothstep(-0.05, 0.55, e));
  float sb = pow(max(dot(d, u_sun), 0.0), 5.0) * 0.45 + pow(max(dot(d, u_sun), 0.0), 40.0) * 0.5;
  vec3 c = vec3(base + sb);
  // the camp backdrop seen in reflections (same image as the billboard)
  if (d.y > 0.05) {
    float t = (u_bd.x - u_eye.y) / d.y;
    vec3 q = u_eye + d * t;
    vec2 uv = vec2((q.x - u_bd.y) / u_bd.z + 0.5, 1.0 - (q.z - u_bd.w) / u_bd_h);
    if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) {
      vec4 b = textureLod(u_backdrop, uv, 2.0);
      c = mix(c, b.rgb / max(b.a, 1e-3), b.a);
    }
  }
  return c;
}

// rain impact field: cells of CELL cm, each cell gets impacts with period P (jittered)
const float CELL = 2.6;
vec4 impact(vec2 cc, float t, float k_off){
  float h1 = hash12(cc * 1.37 + 11.0 + k_off);
  float P = 0.42 + 0.55 * h1;
  float tt = t + h1 * P * 7.0;
  float k = floor(tt / P);
  float age = tt - k * P;
  vec2 pos = (cc + hash22(cc + vec2(k * 1.31, k * 2.71) + k_off)) * CELL;
  return vec4(pos, age, k);
}
vec2 ripples(vec2 p, float t){
  vec2 g = vec2(0.0);
  vec2 c = floor(p / CELL);
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    vec2 cc = c + vec2(i, j);
    for (int l = 0; l < 2; l++) {
      vec4 im = impact(cc, t - float(l) * 0.0, float(l) * 17.0);
      float age = im.z;
      vec2 d = p - im.xy; float r = length(d) + 1e-4;
      float R = age * 14.0;
      float x = r - R;
      float sg = 0.55 + age * 1.2;
      float env = exp(-age * 2.6) * exp(-x * x / (sg * sg));
      float kk = 5.5;
      float dh = env * (kk * cos(kk * x) - 2.0 * x / (sg * sg) * sin(kk * x));
      g += dh * d / r * 0.05 * smoothstep(0.0, 0.03, age);
    }
  }
  return g * u_rain;
}
float ggx(vec3 n, vec3 v, vec3 l, float rough){
  vec3 h = normalize(v + l); float a = rough * rough; float nh = max(dot(n, h), 0.0);
  float d = a * a / (3.14159 * pow(nh * nh * (a * a - 1.0) + 1.0, 2.0));
  return d;
}
"""

TERRAIN_VS = COMMON + r"""
in vec2 in_xy;
uniform mat4 u_vp;
uniform float u_far_mesh;
out vec3 v_pos;
void main(){
  float z = Hgt(in_xy);
  if (u_far_mesh > 0.5) z -= 0.02;
  // the soaked paper presses a shallow bed into the soft mud
  vec4 tm = tmap(in_xy, 1.5);
  z -= 0.45 * clamp(tm.r * 1.4, 0.0, 1.0) * (1.0 - smoothstep(0.3, 2.0, tm.g / max(tm.r, 1e-3)));
  v_pos = vec3(in_xy, z);
  gl_Position = u_vp * vec4(v_pos, 1.0);
}
"""

TERRAIN_FS = COMMON + r"""
in vec3 v_pos;
uniform float u_far_mesh;
uniform mat4 u_view;
layout(location=0) out vec4 o_col;
layout(location=1) out vec4 o_aux;   // r = linear depth, g = flag mask
void main(){
  vec2 p = v_pos.xy;
  if (u_far_mesh > 0.5) {
    vec2 un = (p - u_near_rect.xy) / u_near_rect.zw;
    if (un.x > 0.02 && un.y > 0.02 && un.x < 0.98 && un.y < 0.98) discard;
  }
  float w = nearW(p);
  float e = mix(u_near_rect.z / 2048.0, u_far_rect.z / 2048.0, w) * 1.0;
  float hx = Hgt(p + vec2(e, 0)) - Hgt(p - vec2(e, 0));
  float hy = Hgt(p + vec2(0, e)) - Hgt(p - vec2(0, e));
  vec3 n = normalize(vec3(-hx / (2.0 * e), -hy / (2.0 * e), 1.0));
  float h = v_pos.z;
  vec3 V = normalize(u_eye - v_pos);
  float dist = length(u_eye - v_pos);
  // micro relief (fades with distance so it never aliases)
  float fw = length(fwidth(p));
  float mfade = 1.0 - smoothstep(0.08, 0.5, fw);
  vec2 mp = p * 3.2;
  float m0 = vnoise(mp), mx = vnoise(mp + vec2(0.05, 0)), my = vnoise(mp + vec2(0, 0.05));
  vec2 mg = vec2(mx - m0, my - m0) / 0.05;
  n = normalize(n + vec3(-mg * 0.012 * mfade, 0.0));
  // mud albedo: wet playa clay, darker in troughs, lighter on crests, grit specks
  float lo = vnoise(p * 0.05) * 0.6 + vnoise(p * 0.21) * 0.4;
  float crest = clamp((h + 0.4) * 0.35, 0.0, 1.0);
  float alb = mix(0.07, 0.15, lo) * mix(0.8, 1.35, crest);
  float grit = step(0.93, hash12(floor(p * 9.0))) * mfade;
  alb *= 1.0 - 0.35 * grit * hash12(floor(p * 9.0) + 3.0);
  // contact shadow + damp halo under/around the paper
  vec4 tm0 = tmap(p, 1.0);
  vec4 tm1 = tmap(p, 4.0);
  float occ = 1.0 - 0.55 * clamp(tm1.r * (1.0 - smoothstep(0.0, 12.0, tm1.g / max(tm1.r, 1e-3))), 0.0, 1.0);
  occ *= 1.0 - 0.35 * tm0.r * (1.0 - smoothstep(0.0, 1.5, tm0.g / max(tm0.r, 1e-3)));
  float water_d = u_water - h;
  vec3 col;
  vec3 L = u_sun;
  if (water_d > 0.0) {
    // rain puddle: murky water, mirror at grazing angles, rippled by the rain
    vec2 rg = ripples(p, u_time);
    vec3 wn = normalize(vec3(-rg, 1.0));
    float fres = 0.02 + 0.98 * pow(1.0 - max(dot(wn, V), 0.0), 5.0);
    vec3 R = reflect(-V, wn);
    vec3 refl = sky(R);
    // occlude the reflection by the paper lying there (cheap: the contact map)
    refl *= mix(1.0, 0.6, tm1.r);
    float depth_k = clamp(water_d / 1.2, 0.0, 1.0);
    vec3 below = vec3(alb * 0.55 * (1.0 - 0.6 * depth_k) + 0.05 * depth_k);
    col = mix(below * (0.7 + 0.3 * n.z), refl, fres);
    col += vec3(ggx(wn, V, L, 0.06) * 0.04);
    // soft shoreline
    float shore = smoothstep(0.0, 0.08, water_d);
    vec3 mudc = vec3(alb) * (0.55 + 0.45 * (0.5 + 0.5 * n.z));
    col = mix(mudc * 0.8, col, shore);
  } else {
    float wet = 1.0;
    // wet (dark, glossy) vs merely damp (lighter, matte) mud, patchy
    float wetm = smoothstep(0.30, 0.70, vnoise(p * 0.06) * 0.6 + vnoise(p * 0.27 + 7.0) * 0.4);
    wetm = max(wetm, 1.0 - smoothstep(0.0, 0.6, h - u_water));      // always wet near puddles
    float rough = mix(0.55, 0.14, wetm) * mix(1.0, 1.3, crest);
    alb *= mix(1.45, 0.85, wetm);
    vec3 diff = vec3(alb) * (0.55 + 0.45 * (0.5 + 0.5 * n.z)) * (0.80 + 0.30 * max(dot(n, L), 0.0));
    vec3 R = reflect(-V, n);
    float nv = max(dot(n, V), 0.0);
    float fres = 0.02 + (max(1.0 - rough, 0.02) - 0.02) * pow(1.0 - nv, 5.0);
    vec3 spec = sky(R) * fres * wet * mix(0.35, 0.8, wetm) + vec3(ggx(n, V, L, rough) * 0.035 * wet);
    // rain hits darken tiny spots then dry
    col = diff + spec;
  }
  col *= occ;
  // rain haze with distance
  float fog = 1.0 - exp(-max(dist - 60.0, 0.0) / 1400.0);
  col = mix(col, vec3(0.64), fog * 0.85);
  vec4 vp = u_view * vec4(v_pos, 1.0);
  o_col = vec4(col, 1.0);
  o_aux = vec4(-vp.z, 0.0, 0.0, 1.0);
}
"""

QUAD_VS = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# backdrop billboard (a big vertical plane far away) — drawn with the same COMMON
BD_VS = COMMON + r"""
in vec2 in_pos;   // 0..1
uniform mat4 u_vp;
out vec2 v_uv; out vec3 v_pos;
void main(){
  v_uv = vec2(in_pos.x, 1.0 - in_pos.y);
  v_pos = vec3(u_bd.y + (in_pos.x - 0.5) * u_bd.z, u_bd.x, u_bd.w + in_pos.y * u_bd_h);
  gl_Position = u_vp * vec4(v_pos, 1.0);
}
"""
BD_FS = COMMON + r"""
in vec2 v_uv; in vec3 v_pos;
uniform mat4 u_view;
layout(location=0) out vec4 o_col;
layout(location=1) out vec4 o_aux;
void main(){
  vec3 d = normalize(v_pos - u_eye);
  vec4 b = texture(u_backdrop, v_uv);
  if (b.a < 0.02) discard;
  vec3 c = b.rgb / b.a;
  // rain haze
  c = mix(c, vec3(0.66), 0.30);
  vec4 vp = u_view * vec4(v_pos, 1.0);
  o_col = vec4(c, 1.0);
  o_aux = vec4(-vp.z, 0.0, 0.0, 1.0);
}
"""

# screen-space layer (flag canvas / people) placed at a given view depth
LAYER_FS = r"""
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_depth;     // linear view depth (cm)
uniform float u_is_flag;
uniform float u_ndc_z;
layout(location=0) out vec4 o_col;
layout(location=1) out vec4 o_aux;
void main(){
  vec4 c = texture(u_tex, v_uv);   // premultiplied sRGB (flag colours as drawn); uploaded flipped
  if (c.a < 0.004) discard;
  vec3 lin = pow(max(c.rgb / c.a, 0.0), vec3(2.2));
  o_col = vec4(lin, c.a);
  o_aux = vec4(u_depth, u_is_flag * c.a, 0.0, c.a);
  gl_FragDepth = c.a > 0.5 ? u_ndc_z : 1.0;
}
"""

TRACT_VS = r"""
#version 330
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_edge;
uniform mat4 u_vp;
out vec3 v_pos; out vec3 v_nrm; out vec2 v_uv; out float v_edge;
void main(){ v_pos = in_pos; v_nrm = in_nrm; v_uv = in_uv; v_edge = in_edge; gl_Position = u_vp * vec4(in_pos, 1.0); }
"""
TRACT_FS = COMMON + r"""
in vec3 v_pos; in vec3 v_nrm; in vec2 v_uv; in float v_edge;
uniform sampler2D u_front;   // rgb = printed page (flags spotted in), a = flag coverage
uniform sampler2D u_back;
uniform vec2 u_page_cm;      // page size (cm)
uniform vec4 u_face;         // x = beads front, y = beads back, z = mud front, w = mud back
uniform vec2 u_seed;
uniform mat4 u_view;
uniform float u_gloss;
layout(location=0) out vec4 o_col;
layout(location=1) out vec4 o_aux;

float beadField(vec2 q, float dens, float seed, out vec2 ctr, out float rad){
  // q in cm on the page; returns signed distance inside the nearest bead (0..1 at centre), else 0
  float cell = 0.42;
  vec2 c = floor(q / cell);
  float best = 0.0; ctr = vec2(0.0); rad = 0.0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec2 cc = c + vec2(i, j);
    float hh = hash12(cc + seed);
    if (hh > dens) continue;
    vec2 pos = (cc + 0.15 + 0.7 * hash22(cc + seed * 1.7)) * cell;
    float r = cell * (0.12 + 0.55 * pow(hash12(cc * 1.9 + seed), 2.5));
    // slightly squashed drops
    vec2 d = q - pos; d.y *= 1.0 + 0.25 * hash12(cc + 4.0);
    float k = 1.0 - length(d) / r;
    if (k > best) { best = k; ctr = pos; rad = r; }
  }
  return best;
}

void main(){
  bool fr = gl_FrontFacing;
  vec2 uv = fr ? v_uv : vec2(1.0 - v_uv.x, v_uv.y);
  vec3 n = normalize(fr ? v_nrm : -v_nrm);
  vec3 V = normalize(u_eye - v_pos);
  vec4 tex = fr ? texture(u_front, uv) : texture(u_back, uv);
  float flagm = clamp(tex.a, 0.0, 1.0);
  // the flag keep-out zone (beads/mud/spec never come near the cloth)
  float keep = fr ? textureLod(u_front, uv, 4.0).a : textureLod(u_back, uv, 4.0).a;
  keep = clamp(max(keep * 6.0, flagm * 4.0), 0.0, 1.0);
  vec3 print = pow(tex.rgb, vec3(2.2));
  float lum = dot(print, vec3(0.2126, 0.7152, 0.0722));
  vec3 base = mix(vec3(lum), print, flagm);            // paper + ink are neutral; only the flag keeps colour
  vec2 q = uv * u_page_cm;
  float beads = fr ? u_face.x : u_face.y;
  float mud = fr ? u_face.z : u_face.w;
  // water beads: magnify the print beneath, dark refraction rim, sky highlight
  vec2 bc; float br;
  float bk = beadField(q, beads, fr ? u_seed.x : u_seed.y, bc, br);
  vec3 bn = n;
  if (bk > 0.0 && keep < 0.02) {
    vec2 buv = (bc + (q - bc) * 0.55) / u_page_cm;
    vec3 bp = pow((fr ? texture(u_front, buv) : texture(u_back, buv)).rgb, vec3(2.2));
    float bl = dot(bp, vec3(0.2126, 0.7152, 0.0722));
    base = mix(base, vec3(bl) * 1.05, smoothstep(0.0, 0.25, bk));
    // cap normal in page space -> world via screen derivatives
    vec3 dpdx = dFdx(v_pos), dpdy = dFdy(v_pos);
    vec2 dqdx = dFdx(q), dqdy = dFdy(q);
    float det = dqdx.x * dqdy.y - dqdx.y * dqdy.x;
    vec3 Tt = (dpdx * dqdy.y - dpdy * dqdx.y) / (abs(det) > 1e-12 ? det : 1e-12);
    vec3 Bt = (dpdy * dqdx.x - dpdx * dqdy.x) / (abs(det) > 1e-12 ? det : 1e-12);
    Tt = normalize(Tt - n * dot(n, Tt)); Bt = normalize(cross(n, Tt)) * sign(dot(cross(n, Tt), Bt));
    vec2 dd = (q - bc) / br;
    float cap = sqrt(max(1.0 - dot(dd, dd), 0.0));
    bn = normalize(Tt * dd.x * 1.4 + Bt * dd.y * 1.4 + n * max(cap, 0.08));
    float rim = smoothstep(0.35, 0.0, bk);
    base *= 1.0 - 0.55 * rim;
  }
  // mud smears/speckles (only on faces that lay in the mud) — never near a flag
  if (mud > 0.0 && keep < 0.02) {
    float edge = v_edge;
    float sm = vnoise(q * 0.9 + u_seed) * 0.7 + vnoise(q * 3.7) * 0.3;
    float sp = step(1.0 - 0.06 * mud, hash12(floor(q * 14.0) + u_seed));
    float amt = smoothstep(0.55, 0.85, sm + edge * 0.35 * mud) * mud * 0.55 + sp * 0.5;
    base = mix(base, vec3(0.12), clamp(amt, 0.0, 0.8));
  }
  // wet paper: slightly grey and translucent (show-through of the other side's print)
  if (keep < 0.99) {
    vec2 ouv = vec2(1.0 - uv.x, uv.y);
    vec4 other = fr ? texture(u_back, ouv) : texture(u_front, ouv);
    float ol = dot(pow(other.rgb, vec3(2.2)), vec3(0.2126, 0.7152, 0.0722));
    vec3 wetp = base * (1.0 - 0.10 * (1.0 - ol)) * 0.93;
    base = mix(wetp, base, keep);
  }
  // lighting: overcast dome + soft key; translucency when backlit
  vec3 L = u_sun;
  float amb = 0.62 + 0.38 * (0.5 + 0.5 * n.z);
  float key = 0.80 + 0.30 * max(dot(n, L), 0.0);
  float back = max(-dot(n, V), 0.0);
  vec3 col = base * amb * key;
  // occlusion from the neighbouring leaves / mud contact (never on flag cloth)
  vec4 tm = tmap(v_pos.xy, 3.0);
  float above = tm.g / max(tm.r, 1e-3);
  float selfocc = 1.0 - 0.25 * tm.r * smoothstep(0.4, 3.0, above - (v_pos.z - Hgt(v_pos.xy)));
  col *= mix(selfocc, 1.0, keep);
  // gloss (wet sheen) + bead highlights, except on the flag
  vec3 R = reflect(-V, bn);
  float fres = 0.03 + 0.97 * pow(1.0 - max(dot(bn, V), 0.0), 5.0);
  vec3 spec = sky(R) * fres * u_gloss + vec3(ggx(bn, V, L, bk > 0.0 ? 0.05 : 0.35) * (bk > 0.0 ? 0.25 : 0.03));
  col += spec * (1.0 - keep);
  // the flag: plain yellow under smooth light only
  vec3 flagc = print * clamp(0.90 + 0.14 * n.z + 0.06 * max(dot(n, L), 0.0), 0.86, 1.06);
  col = mix(col, flagc, flagm);
  vec4 vp = u_view * vec4(v_pos, 1.0);
  o_col = vec4(col, 1.0);
  o_aux = vec4(-vp.z, flagm, 0.0, 1.0);
}
"""

TMAP_VS = r"""
#version 330
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_edge;
uniform vec4 u_tmap_rect;
uniform sampler2D u_hnear;
uniform vec4 u_near_rect;
out float v_h;
void main(){
  vec2 uv = (in_pos.xy - u_tmap_rect.xy) / u_tmap_rect.zw;
  float g = texture(u_hnear, (in_pos.xy - u_near_rect.xy) / u_near_rect.zw).r;
  v_h = max(in_pos.z - g, 0.0);
  gl_Position = vec4(uv * 2.0 - 1.0, 0.5 - 0.001 * in_pos.z, 1.0);
}
"""
TMAP_FS = r"""
#version 330
in float v_h;
out vec4 o;
void main(){ o = vec4(1.0, v_h, 0.0, 1.0); }
"""

TILE_FS = r"""
#version 330
in vec2 v_uv;
uniform sampler2D u_aux;
uniform vec2 u_px;        // 1/full size
uniform float u_focus; uniform float u_K; uniform float u_maxc;
out vec4 o;
float coc(float z){ return min(u_K * abs(1.0 - u_focus / max(z, 0.1)), u_maxc); }
void main(){
  // this tile covers TILE x TILE full-res pixels
  float m = 0.0;
  vec2 base = v_uv;
  for (int j = -8; j < 8; j++) for (int i = -8; i < 8; i++) {
    float z = texture(u_aux, base + vec2(float(i) + 0.5, float(j) + 0.5) * u_px).r;
    m = max(m, coc(z));
  }
  o = vec4(m, 0.0, 0.0, 1.0);
}
"""

DOF_FS = r"""
#version 330
in vec2 v_uv;
uniform sampler2D u_col;
uniform sampler2D u_aux;
uniform sampler2D u_tile;
uniform vec2 u_tpx;       // 1/tile size
uniform vec2 u_px;        // 1/size
uniform float u_focus;
uniform float u_K;        // CoC px at infinity
uniform float u_maxc;     // max CoC px
uniform float u_dbg;
layout(location=0) out vec4 o_col;
layout(location=1) out vec4 o_aux;
float coc(float z){ return min(u_K * abs(1.0 - u_focus / max(z, 0.1)), u_maxc); }
void main(){
  vec4 a0 = texture(u_aux, v_uv);
  float z0 = a0.r;
  float c0 = coc(z0);
  float R = c0;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) R = max(R, texture(u_tile, v_uv + vec2(i, j) * u_tpx).r);
  R = max(R, 0.5);
  float w0 = 1.0 / max(c0 * c0, 1.0);
  vec3 acc = texture(u_col, v_uv).rgb * w0; float fm = a0.g * w0; float wsum = w0;
  const int N = 80;
  const float GA = 2.39996323;
  for (int i = 0; i < N; i++) {
    float r = sqrt((float(i) + 0.5) / float(N)) * R;
    float th = float(i) * GA;
    vec2 uv = v_uv + vec2(cos(th), sin(th)) * r * u_px;
    vec4 a = texture(u_aux, uv);
    float cs = coc(a.r);
    float ce = a.r > z0 ? min(cs, c0 + 1.0) : cs;
    float spacing = R / sqrt(float(N));
    float w = clamp((ce - r) / max(spacing, 0.5) + 1.0, 0.0, 1.0) / max(ce * ce, 1.0);
    acc += texture(u_col, uv).rgb * w; fm += a.g * w; wsum += w;
  }
  o_col = vec4(acc / wsum, 1.0); if (u_dbg > 0.5) o_col = vec4(texture(u_col, v_uv).rgb, 1.0);
  o_aux = vec4(z0, fm / wsum, c0, 1.0);
}
"""

RAIN_VS = COMMON + r"""
in float in_id;
uniform mat4 u_vp;
uniform mat4 u_view;
uniform vec3 u_box;        // box size (cm) around the camera
uniform vec3 u_vel;        // fall velocity cm/s
uniform float u_shutter;
uniform vec2 u_res;
uniform float u_K; uniform float u_focus; uniform float u_maxc;
uniform vec3 u_boxc;
out float v_a; out vec2 v_q; out float v_len; out float v_w;
void main(){
  int id = int(in_id) / 6; int corner = int(in_id) % 6;
  vec2 hA = hash22(vec2(float(id) * 0.123, 7.1)); vec2 hB = hash22(vec2(float(id) * 0.917, 3.3));
  vec3 p0 = vec3(hA.x, hA.y, hB.x) * u_box;
  float sp = 0.85 + 0.3 * hB.y;
  vec3 p = u_boxc + mod(p0 + u_vel * sp * u_time - u_boxc + u_box * 0.5, u_box) - u_box * 0.5;
  vec3 tail = p - u_vel * sp * u_shutter;
  float g = Hgt(p.xy);
  float cut = 0.0;
  if (p.z < g) { p.z = g; cut = 1.0; }
  if (tail.z < g) tail.z = g;
  vec4 c0 = u_vp * vec4(p, 1.0); vec4 c1 = u_vp * vec4(tail, 1.0);
  float z = -(u_view * vec4(p, 1.0)).z;
  if (c0.w < 1.0 || c1.w < 1.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); v_a = 0.0; return; }
  vec2 s0 = c0.xy / c0.w, s1 = c1.xy / c1.w;
  float cz = min(u_K * abs(1.0 - u_focus / z), u_maxc);
  float wpx = 0.9 + 0.035 * 540.0 / z * 3.0 + cz;       // drop width + blur (px at 1080p)
  vec2 dir = s1 - s0; vec2 dpx = dir * u_res * 0.5; float lpx = length(dpx) + 1e-3;
  vec2 dn = dpx / lpx; vec2 nn = vec2(-dn.y, dn.x);
  float lenpx = lpx + cz;
  vec2 ctr = (s0 + s1) * 0.5 * u_res * 0.5;
  float side = (corner == 0 || corner == 2 || corner == 3) ? -1.0 : 1.0;
  float end = (corner == 0 || corner == 1 || corner == 4) ? -1.0 : 1.0;
  if (corner == 5) { side = -1.0; end = 1.0; }
  if (corner == 3) { side = 1.0; end = -1.0; }
  vec2 q = vec2(end, side);
  vec2 ppx = ctr + dn * (end * lenpx * 0.5) + nn * (side * wpx);
  gl_Position = vec4(ppx / (u_res * 0.5), c0.z / c0.w, 1.0);
  v_q = q; v_len = lenpx; v_w = wpx;
  float bright = 0.10 + 0.16 * hA.y * hA.y;
  float defocus = clamp(1.6 / wpx, 0.0, 1.0);
  v_a = bright * defocus * u_rain * (1.0 - cut * 0.5) * smoothstep(6.0, 16.0, z);
}
"""
RAIN_FS = r"""
#version 330
in float v_a; in vec2 v_q; in float v_len; in float v_w;
uniform sampler2D u_aux;   // scene linear depth (for occlusion)
uniform vec2 u_px;
out vec4 o;
void main(){
  float across = 1.0 - abs(v_q.y);
  float along = 1.0 - smoothstep(0.6, 1.0, abs(v_q.x));
  float a = v_a * across * across * along;
  o = vec4(vec3(a), a);
}
"""

SPLASH_VS = COMMON + r"""
in float in_id;
uniform mat4 u_vp; uniform mat4 u_view;
uniform vec4 u_area;   // x0, y0, w, h (cm) of the splash field
uniform vec2 u_res;
uniform float u_K; uniform float u_focus; uniform float u_maxc;
out float v_a; out vec2 v_q;
void main(){
  int id = int(in_id) / 6; int corner = int(in_id) % 6;
  int per = 6;
  int cell_id = id / per; int part = id % per;
  int nx = int(u_area.z / CELL);
  vec2 cc = floor(u_area.xy / CELL) + vec2(float(cell_id % nx), float(cell_id / nx));
  vec4 im = impact(cc, u_time, 0.0);
  float age = im.z;
  float life = 0.16;
  v_a = 0.0;
  gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
  if (age > life) return;
  if (hash12(cc * 0.71 + im.w * 3.3) > 0.22) return;
  vec2 h = hash22(cc * 3.1 + float(part) * 1.7 + im.w);
  float ang = h.x * 6.2832; float spd = 18.0 + 30.0 * h.y;
  vec3 v0 = vec3(cos(ang) * spd, sin(ang) * spd, 55.0 + 55.0 * h.y);
  vec4 tm = tmap(im.xy, 0.0);
  float g = max(Hgt(im.xy), Hgt(im.xy) + tm.g * step(0.5, tm.r));
  if (u_water > Hgt(im.xy)) g = max(g, u_water);
  vec3 p = vec3(im.xy, g) + v0 * age + vec3(0.0, 0.0, -490.0) * age * age;
  if (p.z < g) return;
  vec4 c = u_vp * vec4(p, 1.0);
  if (c.w < 1.0) return;
  float z = -(u_view * vec4(p, 1.0)).z;
  float cz = min(u_K * abs(1.0 - u_focus / z), u_maxc);
  float r = 0.5 + 0.035 * 540.0 / z * 2.0 + cz;
  vec2 q = vec2((corner == 1 || corner == 2 || corner == 4) ? 1.0 : -1.0, (corner == 2 || corner == 4 || corner == 5) ? 1.0 : -1.0);
  vec2 s = c.xy / c.w + q * r / (u_res * 0.5);
  gl_Position = vec4(s, c.z / c.w, 1.0);
  v_q = q;
  v_a = 0.30 * (1.0 - age / life) * pow(clamp(1.2 / r, 0.0, 1.0), 2.5) * u_rain;
}
"""
SPLASH_FS = r"""
#version 330
in float v_a; in vec2 v_q;
out vec4 o;
void main(){ float d = 1.0 - length(v_q); if (d <= 0.0) discard; float a = v_a * smoothstep(0.0, 0.6, d); o = vec4(vec3(a), a); }
"""

FINAL_FS = r"""
#version 330
in vec2 v_uv;
uniform sampler2D u_col;   // DOF colour (linear)
uniform sampler2D u_aux;   // DOF aux (g = flag mask)
uniform sampler2D u_rain;  // additive rain
uniform float u_expo;
out vec4 o;
void main(){
  vec3 c = texture(u_col, v_uv).rgb;
  float fm = texture(u_aux, v_uv).g;
  vec4 r = texture(u_rain, v_uv);
  float keep = 1.0 - clamp(fm * 2.5, 0.0, 1.0);              // rain NEVER over a flag
  // a raindrop brightens dark mud and slightly veils bright paper
  vec3 lum = vec3(dot(c, vec3(0.2126, 0.7152, 0.0722)));
  c = c + r.rgb * keep * (0.65 - 0.35 * lum);
  c *= u_expo;
  // filmic shoulder, then sRGB
  c = c / (1.0 + 0.10 * c);
  c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
  o = vec4(c, 1.0);
}
"""


def _gl():
    pid = os.getpid()
    if _CTX.get("pid") != pid:
        import moderngl
        _CTX.clear()
        _CTX["pid"] = pid
        _CTX["ctx"] = moderngl.create_standalone_context(backend="egl")
    return _CTX["ctx"]


class World:
    """GL resources for one process at one pixel size."""

    def __init__(self, mud, w, h, backdrop_rgba):
        import moderngl
        self.mgl = moderngl
        ctx = self.ctx = _gl()
        self.w, self.h = w, h
        self.mud = mud
        # heightfields
        self.t_near = ctx.texture(mud["near"].shape[::-1], 1, mud["near"].astype("f4").tobytes(), dtype="f4")
        self.t_far = ctx.texture(mud["far"].shape[::-1], 1, mud["far"].astype("f4").tobytes(), dtype="f4")
        for t in (self.t_near, self.t_far):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        # backdrop
        bd = np.ascontiguousarray(np.flipud(backdrop_rgba).astype("f4"))
        self.t_bd = ctx.texture((bd.shape[1], bd.shape[0]), 4, np.flipud(bd).tobytes(), dtype="f4")
        self.t_bd.build_mipmaps()
        self.t_bd.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.t_bd.repeat_x = self.t_bd.repeat_y = False
        # programs
        self.p_terr = ctx.program(vertex_shader=TERRAIN_VS, fragment_shader=TERRAIN_FS)
        self.p_bd = ctx.program(vertex_shader=BD_VS, fragment_shader=BD_FS)
        self.p_layer = ctx.program(vertex_shader=QUAD_VS, fragment_shader=LAYER_FS)
        self.p_tract = ctx.program(vertex_shader=TRACT_VS, fragment_shader=TRACT_FS)
        self.p_tmap = ctx.program(vertex_shader=TMAP_VS, fragment_shader=TMAP_FS)
        self.p_dof = ctx.program(vertex_shader=QUAD_VS, fragment_shader=DOF_FS)
        self.p_tile = ctx.program(vertex_shader=QUAD_VS, fragment_shader=TILE_FS)
        self.p_rain = ctx.program(vertex_shader=RAIN_VS, fragment_shader=RAIN_FS)
        self.p_splash = ctx.program(vertex_shader=SPLASH_VS, fragment_shader=SPLASH_FS)
        self.p_final = ctx.program(vertex_shader=QUAD_VS, fragment_shader=FINAL_FS)
        # meshes
        self.vao_near = self._grid_vao(mud["near_rect"], 700, 0.0)
        self.vao_far = self._grid_vao(mud["far_rect"], 700, 1.0)
        quad = np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], "f4")
        self.vbo_quad = ctx.buffer(quad.tobytes())
        self.vao_quads = {p: ctx.vertex_array(p, [(self.vbo_quad, "2f", "in_pos")]) for p in (self.p_layer, self.p_dof, self.p_final, self.p_tile)}
        q01 = np.array([0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1], "f4")
        self.vbo_q01 = ctx.buffer(q01.tobytes())
        self.vao_bd = ctx.vertex_array(self.p_bd, [(self.vbo_q01, "2f", "in_pos")])
        self.n_rain = 3600
        self.vbo_rain = ctx.buffer(np.arange(self.n_rain * 6, dtype="f4").tobytes())
        self.vao_rain = ctx.vertex_array(self.p_rain, [(self.vbo_rain, "1f", "in_id")])
        self.splash_area = (-70.0, -30.0, 140.0, 120.0)
        ncell = int(self.splash_area[2] / 2.6) * int(self.splash_area[3] / 2.6)
        self.n_splash = ncell * 6
        self.vbo_splash = ctx.buffer(np.arange(self.n_splash * 6, dtype="f4").tobytes())
        self.vao_splash = ctx.vertex_array(self.p_splash, [(self.vbo_splash, "1f", "in_id")])
        # framebuffers
        S = 4
        self.rb_col = ctx.renderbuffer((w, h), 4, samples=S, dtype="f2")
        self.rb_aux = ctx.renderbuffer((w, h), 4, samples=S, dtype="f4")
        self.rb_dep = ctx.depth_renderbuffer((w, h), samples=S)
        self.fbo_ms = ctx.framebuffer([self.rb_col, self.rb_aux], self.rb_dep)
        self.t_col = ctx.texture((w, h), 4, dtype="f2")
        self.t_aux = ctx.texture((w, h), 4, dtype="f4")
        self.t_aux.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo_res = ctx.framebuffer([self.t_col, self.t_aux])
        self.t_dcol = ctx.texture((w, h), 4, dtype="f2")
        self.t_daux = ctx.texture((w, h), 4, dtype="f4")
        self.fbo_dof = ctx.framebuffer([self.t_dcol, self.t_daux])
        self.tw, self.th = (w + 15) // 16, (h + 15) // 16
        self.t_tile = ctx.texture((self.tw, self.th), 4, dtype="f4")
        self.t_tile.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo_tile = ctx.framebuffer([self.t_tile])
        self.t_rain = ctx.texture((w, h), 4, dtype="f2")
        self.fbo_rain = ctx.framebuffer([self.t_rain])
        self.t_out = ctx.texture((w, h), 4, dtype="f4")
        self.fbo_out = ctx.framebuffer([self.t_out])
        # top-down tract map
        self.tmap_n = 512
        self.t_tmap = ctx.texture((self.tmap_n, self.tmap_n), 4, dtype="f4")
        self.t_tmap.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.fbo_tmap = ctx.framebuffer([self.t_tmap])
        self.tmap_rect = (-70.0, -30.0, 120.0, 120.0)
        self.page_tex = {}
        self.layer_tex = {}

    # ------------------------------------------------------------------ resources
    def _grid_vao(self, rect, n, far):
        x0, y0, sw, sh = rect
        xs = np.linspace(x0, x0 + sw, n, dtype="f4")
        ys = np.linspace(y0, y0 + sh, n, dtype="f4")
        X, Y = np.meshgrid(xs, ys)
        v = np.stack([X, Y], -1).reshape(-1, 2).astype("f4")
        i = np.arange(n * n).reshape(n, n)
        a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
        idx = np.stack([a, b, c, b, d, c], -1).reshape(-1).astype("i4")
        vbo = self.ctx.buffer(v.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        prog = self.p_terr
        return self.ctx.vertex_array(prog, [(vbo, "2f", "in_xy")], ibo)

    def texture(self, key, rgba, dynamic=False):
        """upload a page texture (h, w, 4) float [0,1] (rgb = print, a = flag coverage). Cached by key."""
        mgl = self.mgl
        h, w = rgba.shape[:2]
        data = np.ascontiguousarray(np.flipud(np.clip(rgba * 255 + 0.5, 0, 255).astype("u1")))
        t = self.page_tex.get(key)
        if t is None or t.size != (w, h):
            t = self.ctx.texture((w, h), 4, data.tobytes())
            t.repeat_x = t.repeat_y = False
            t.anisotropy = 16.0
            self.page_tex[key] = t
        else:
            t.write(data.tobytes())
        t.build_mipmaps()
        t.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        return t

    def _set(self, prog, **kw):
        for k, v in kw.items():
            if k in prog:
                u = prog[k]
                if isinstance(v, np.ndarray) and v.shape == (4, 4):
                    u.write(v.T.astype("f4").tobytes())
                else:
                    u.value = v

    # ------------------------------------------------------------------ frame
    def render(self, cam, T, leaves, layers=(), rain=1.0, expo=1.0, sun=(-0.35, 0.55, 0.75), s=1.0, splash=True,
               rain_vel=(-190.0, 60.0, -900.0), gloss=0.55):
        """leaves: list of dict(pos (ny,nx,3) cm, uv (ny,nx,2), front=texkey, back=texkey, face=(bf,bb,mf,mb),
        page_cm=(w,h), seed=(a,b)); layers: list of dict(rgba (h,w,4) premult in screen px, depth cm, flag bool)
        -> float32 (h, w, 3) sRGB"""
        mgl = self.mgl
        ctx = self.ctx
        V = cam.view()
        P = perspective(cam.fovy, self.w / self.h)
        VP = P @ V
        sun = np.asarray(sun, np.float64)
        sun = tuple((sun / np.linalg.norm(sun)).tolist())
        nr = self.mud["near_rect"]
        fr_ = self.mud["far_rect"]
        common = dict(u_near_rect=tuple(nr), u_far_rect=tuple(fr_), u_water=float(self.mud["water"]), u_time=float(T),
                      u_eye=tuple(cam.eye.tolist()), u_sun=sun, u_bd=(2650.0, -80.0, 6400.0, -300.0), u_bd_h=1500.0,
                      u_tmap_rect=self.tmap_rect, u_rain=float(rain), u_hnear=0, u_hfar=1, u_backdrop=2, u_tmap=3)
        self.t_near.use(0)
        self.t_far.use(1)
        self.t_bd.use(2)

        # ---- 1. top-down tract map
        vaos = []
        for lf in leaves:
            vaos.append(self._leaf_vao(lf))
        self.fbo_tmap.use()
        ctx.viewport = (0, 0, self.tmap_n, self.tmap_n)
        self.fbo_tmap.clear(0, 0, 0, 0)
        ctx.disable(mgl.DEPTH_TEST | mgl.CULL_FACE)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE)
        ctx.blend_equation = mgl.MAX
        self._set(self.p_tmap, u_tmap_rect=self.tmap_rect, u_near_rect=tuple(nr), u_hnear=0)
        for vao_t, _ in vaos:
            vao_t.render()
        ctx.blend_equation = mgl.FUNC_ADD
        self.t_tmap.build_mipmaps()
        self.t_tmap.use(3)

        # ---- 2. scene
        self.fbo_ms.use()
        ctx.viewport = (0, 0, self.w, self.h)
        self.fbo_ms.clear(0.6, 0.6, 0.6, 1.0, depth=1.0)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.BLEND)
        ctx.disable(mgl.CULL_FACE)
        for prog in (self.p_terr, self.p_bd):
            self._set(prog, u_vp=VP, u_view=V, **common)
        self.vao_bd.render()
        self._set(self.p_terr, u_far_mesh=1.0)
        self.vao_far.render()
        self._set(self.p_terr, u_far_mesh=0.0)
        self.vao_near.render()
        # screen layers (flag canvas, people) at their depth
        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        for k, L in enumerate(layers):
            t = self._layer_tex(k, L["rgba"])
            t.use(5)
            z = float(L["depth"])
            clip = P @ np.array([0, 0, -z, 1.0])
            self._set(self.p_layer, u_tex=5, u_depth=z, u_is_flag=1.0 if L.get("flag") else 0.0,
                      u_ndc_z=float((clip[2] / clip[3]) * 0.5 + 0.5))
            self.vao_quads[self.p_layer].render()
        ctx.disable(mgl.BLEND)
        # the tract
        self._set(self.p_tract, u_vp=VP, u_view=V, u_gloss=float(gloss), **common)
        for (vao_t, vao_s), lf in zip(vaos, leaves):
            self.page_tex[lf["front"]].use(6)
            self.page_tex[lf["back"]].use(7)
            self._set(self.p_tract, u_front=6, u_back=7, u_page_cm=tuple(lf["page_cm"]), u_face=tuple(lf["face"]),
                      u_seed=tuple(lf.get("seed", (1.0, 2.0))))
            vao_s.render()
        for vao_t, vao_s in vaos:
            vao_t.release()
            vao_s.release()
        for b in self._tmp_bufs:
            b.release()
        self._tmp_bufs = []
        ctx.copy_framebuffer(self.fbo_res, self.fbo_ms)

        # ---- 3. depth of field
        K = cam.blur * s
        maxc = min(48.0 * s, K * 1.3 + 1.0)
        ctx.disable(mgl.DEPTH_TEST)
        self.t_col.use(0)
        self.t_aux.use(1)
        self.fbo_tile.use()
        ctx.viewport = (0, 0, self.tw, self.th)
        self._set(self.p_tile, u_aux=1, u_px=(1.0 / self.w, 1.0 / self.h), u_focus=float(cam.focus), u_K=float(K),
                  u_maxc=float(maxc))
        self.vao_quads[self.p_tile].render()
        self.t_tile.use(2)
        self.fbo_dof.use()
        ctx.viewport = (0, 0, self.w, self.h)
        self._set(self.p_dof, u_col=0, u_aux=1, u_tile=2, u_tpx=(1.0 / self.tw, 1.0 / self.th),
                  u_px=(1.0 / self.w, 1.0 / self.h), u_focus=float(cam.focus), u_K=float(K), u_maxc=float(maxc), u_dbg=float(os.environ.get('T01DBG', 0)))
        self.vao_quads[self.p_dof].render()

        # ---- 4. rain + splashes
        self.fbo_rain.use()
        self.fbo_rain.clear(0, 0, 0, 0)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = (mgl.ONE, mgl.ONE)
        self.t_near.use(0)
        self.t_far.use(1)
        self.t_bd.use(2)
        self.t_tmap.use(3)
        self.t_aux.use(4)
        fwd = cam.target - cam.eye
        fwd = fwd / np.linalg.norm(fwd)
        boxc = cam.eye + fwd * 70.0
        rain_common = dict(common)
        rain_common.update(u_vp=VP, u_view=V, u_res=(float(self.w), float(self.h)), u_K=float(K), u_focus=float(cam.focus),
                           u_maxc=float(maxc))
        self._set(self.p_rain, u_box=(160.0, 160.0, 120.0), u_vel=tuple(rain_vel), u_shutter=1.0 / 48.0,
                  u_boxc=tuple(boxc.tolist()), u_aux=4, u_px=(1.0 / self.w, 1.0 / self.h), **rain_common)
        if rain > 0:
            # manual depth test against the resolved scene depth: use the MS depth via a depth-only copy
            self.vao_rain.render(mgl.TRIANGLES)
            if splash:
                self._set(self.p_splash, u_area=self.splash_area, **rain_common)
                self.vao_splash.render(mgl.TRIANGLES)
        ctx.disable(mgl.BLEND)

        # ---- 5. final
        self.fbo_out.use()
        self.t_dcol.use(0)
        self.t_daux.use(1)
        self.t_rain.use(2)
        self._set(self.p_final, u_col=0, u_aux=1, u_rain=2, u_expo=float(expo))
        self.vao_quads[self.p_final].render()
        img = np.frombuffer(self.fbo_out.read(components=4, dtype="f4"), "f4").reshape(self.h, self.w, 4)
        return np.ascontiguousarray(np.flipud(img[..., :3]))

    _tmp_bufs = []

    def _leaf_vao(self, lf):
        pos = np.asarray(lf["pos"], "f4")
        ny, nx = pos.shape[:2]
        # normals from the grid
        du = np.gradient(pos, axis=1)
        dv = np.gradient(pos, axis=0)
        nrm = np.cross(du, dv)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
        uv = np.asarray(lf["uv"], "f4")
        edge = np.minimum(np.minimum(uv[..., 0], 1 - uv[..., 0]), np.minimum(uv[..., 1], 1 - uv[..., 1]))
        edge = np.clip(1.0 - edge / 0.12, 0, 1)[..., None].astype("f4")
        data = np.concatenate([pos, nrm.astype("f4"), uv, edge], -1).reshape(-1, 9).astype("f4")
        i = np.arange(nx * ny).reshape(ny, nx)
        a, b, c, d = i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]
        idx = np.stack([a, b, c, b, d, c], -1).reshape(-1).astype("i4")
        vbo = self.ctx.buffer(data.tobytes())
        ibo = self.ctx.buffer(idx.tobytes())
        self._tmp_bufs = self._tmp_bufs + [vbo, ibo]
        fmt = [(vbo, "3f 3f 2f 1f", "in_pos", "in_nrm", "in_uv", "in_edge")]
        return (self.ctx.vertex_array(self.p_tmap, [(vbo, "3f 12x 8x 4x", "in_pos")], ibo),
                self.ctx.vertex_array(self.p_tract, fmt, ibo))

    def _layer_tex(self, k, rgba):
        h, w = rgba.shape[:2]
        data = np.ascontiguousarray(np.flipud(np.clip(rgba, 0, 1)).astype("f4"))
        t = self.layer_tex.get(k)
        if t is None or t.size != (w, h):
            t = self.ctx.texture((w, h), 4, dtype="f4")
            t.filter = (self.mgl.LINEAR, self.mgl.LINEAR)
            self.layer_tex[k] = t
        t.write(data.tobytes())
        return t


def get_world(mud, w, h, backdrop_rgba):
    key = ("world", w, h)
    if _CTX.get("pid") == os.getpid() and key in _CTX:
        return _CTX[key]
    _gl()
    wd = World(mud, w, h, backdrop_rgba)
    _CTX[key] = wd
    return wd
