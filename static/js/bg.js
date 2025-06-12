// static/js/bg.js

/** 
 * Create a low-resolution offscreen canvas (windowWidth*RESOLUTION by windowHeight*RESOLUTION),
 * draw noise and color on it, then stretch it to the main canvas.
 * With a CSS blur effect, the result remains soft and smooth.
 */
const RESOLUTION = 0.1;  // Offscreen canvas size ratio (10%)
let off;                 // p5 Graphics offscreen buffer

function setup() {
  pixelDensity(1);      // Disable Retina doubling
  createCanvas(windowWidth, windowHeight);
  noStroke();
  frameRate(15);        // 15fps is smooth enough

  // Initialize offscreen buffer
  const w = max(1, floor(windowWidth * RESOLUTION));
  const h = max(1, floor(windowHeight * RESOLUTION));
  off = createGraphics(w, h);
  off.noStroke();
}

function draw() {
  const palette = [
    '#F5E4D7',
    '#E07A5F',
    '#F4B0A8',
    '#392E2B',
    '#F2CC8F'
  ];
  const t = millis() * 0.0002;

  // Draw pixels on offscreen buffer
  off.loadPixels();
  for (let y = 0; y < off.height; y++) {
    for (let x = 0; x < off.width; x++) {
      let v = noise(x * 0.005, y * 0.005, t);
      let idx = floor(v * palette.length);
      idx = constrain(idx, 0, palette.length - 1);
      const c = color(palette[idx]);

      const i = 4 * (x + y * off.width);
      off.pixels[i + 0] = red(c);
      off.pixels[i + 1] = green(c);
      off.pixels[i + 2] = blue(c);
      off.pixels[i + 3] = 255;
    }
  }
  off.updatePixels();

  // Stretch to main canvas
  image(off, 0, 0, width, height);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  // Recreate offscreen buffer on resize
  const w = max(1, floor(windowWidth * RESOLUTION));
  const h = max(1, floor(windowHeight * RESOLUTION));
  off.remove();
  off = createGraphics(w, h);
  off.noStroke();
}
