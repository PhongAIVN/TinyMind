import React, { useEffect, useRef } from 'react';

const ShaderCanvas: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let animationFrameId: number;
    let gl: WebGLRenderingContext | null = null;
    let program: WebGLProgram | null = null;
    
    // Sync the WebGL drawing-buffer size with the CSS-driven layout size.
    const syncSize = () => {
      const w = canvas.clientWidth || window.innerWidth;
      const h = canvas.clientHeight || window.innerHeight;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
        if (gl) {
          gl.viewport(0, 0, w, h);
        }
      }
    };

    const resizeObserver = new ResizeObserver(syncSize);
    resizeObserver.observe(canvas);
    syncSize();

    gl = canvas.getContext('webgl') || (canvas.getContext('experimental-webgl') as WebGLRenderingContext);
    if (!gl) return;

    const vs = `attribute vec2 a_position;
varying vec2 v_texCoord;
void main() {
  v_texCoord = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

    const fs = `precision highp float;

varying vec2 v_texCoord;

uniform float u_time;
uniform vec2 u_resolution;

// Simplex 2D noise
vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 a0 = x - floor(x + 0.5);
  vec3 m0 = 1.0 - 1.5*(a0*a0 + h*h);
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

void main() {
    // Correct for aspect ratio
    float aspect = u_resolution.x / u_resolution.y;
    vec2 uv = v_texCoord;
    vec2 p = uv * 2.0 - 1.0; // -1 to 1
    p.x *= aspect;
    
    // Time variable for slower, more organic fluid animation
    float t = u_time * 0.12;
    
    // Create multiple layers of noise for fluid motion
    float n1 = snoise(p * 1.0 + vec2(t * 0.5, t * 0.3));
    float n2 = snoise(p * 1.5 - vec2(t * 0.4, t * 0.6));
    float n3 = snoise(p * 2.0 + vec2(t * 0.2, -t * 0.5));
    
    // Combine noises to create flowing warp
    float warp = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;
    
    // Base colors (7 colors from Gemini spectrum)
    vec3 c1 = vec3(0.25, 0.52, 0.96); // Blue
    vec3 c2 = vec3(0.20, 0.80, 0.90); // Cyan
    vec3 c3 = vec3(0.50, 0.20, 0.90); // Purple
    vec3 c4 = vec3(0.90, 0.20, 0.60); // Magenta
    vec3 c5 = vec3(0.95, 0.30, 0.20); // Red
    vec3 c6 = vec3(1.00, 0.80, 0.10); // Yellow
    vec3 c7 = vec3(0.20, 0.70, 0.40); // Green
    
    // Mix them fluidly based on position and warp
    vec3 col = c1;
    col = mix(col, c2, smoothstep(-0.8, 0.8, p.x + warp));
    col = mix(col, c3, smoothstep(-0.8, 0.8, p.y - warp));
    col = mix(col, c4, smoothstep(-0.5, 0.5, sin(p.x * 2.0 + t) * n1));
    col = mix(col, c5, smoothstep(-0.5, 0.5, cos(p.y * 2.0 - t) * n2));
    col = mix(col, c6, smoothstep(0.0, 1.0, n3 + warp));
    col = mix(col, c7, smoothstep(0.2, 0.8, length(p) + warp * 0.5));
    
    // Create a perfect circular soft mask to avoid rectangle edges
    float dist = length(p);
    float mask = smoothstep(1.5, 0.2, dist);
    
    // Add shimmering intensity
    float intensity = 1.0 + 0.15 * sin(t * 2.0 + dist * 5.0);
    col *= intensity;
    
    // Premultiply alpha to prevent black/grey edges when composited by browser
    float alpha = mask * 0.85;
    gl_FragColor = vec4(col * alpha, alpha);
}`;

    function createShader(type: number, src: string) {
      if (!gl) return null;
      const s = gl.createShader(type);
      if (!s) return null;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      return s;
    }

    program = gl.createProgram();
    if (!program) return;
    
    const vShader = createShader(gl.VERTEX_SHADER, vs);
    const fShader = createShader(gl.FRAGMENT_SHADER, fs);
    
    if (vShader) gl.attachShader(program, vShader);
    if (fShader) gl.attachShader(program, fShader);
    
    gl.linkProgram(program);
    gl.useProgram(program);
    
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
    
    const pos = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);
    
    const uTime = gl.getUniformLocation(program, 'u_time');
    const uRes = gl.getUniformLocation(program, 'u_resolution');

    const render = (t: number) => {
      if (!gl) return;
      gl.clearColor(0.0, 0.0, 0.0, 0.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      
      if (uTime) gl.uniform1f(uTime, t * 0.001);
      if (uRes) gl.uniform2f(uRes, canvas.width, canvas.height);
      
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <canvas 
      ref={canvasRef} 
      style={{ display: 'block', width: '100%', height: '100%' }}
    />
  );
};

export default ShaderCanvas;
