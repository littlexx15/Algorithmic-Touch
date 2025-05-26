// static/js/bg.js

/** 
 * 生成一个低分辨率的离屏画布（宽高均为 windowWidth*RESOLUTION、windowHeight*RESOLUTION） 
 * 然后在离屏上跑噪声、上色，再拉伸到主画布。配合 CSS blur 效果，视觉上依然柔和。
 */
const RESOLUTION = 0.1;  // 离屏画布尺寸占比 (0.1 = 10%)
let off;                 // P5 Graphics (离屏画布)

function setup() {
  pixelDensity(1);      // 关闭 Retina 二倍像素
  createCanvas(windowWidth, windowHeight);
  noStroke();
  frameRate(15);        // 15fps 就很流畅了

  // 初始化离屏画布
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

  // 在离屏上操作像素
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

  // 拉伸到主画布
  image(off, 0, 0, width, height);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  // 离屏一起重算
  const w = max(1, floor(windowWidth * RESOLUTION));
  const h = max(1, floor(windowHeight * RESOLUTION));
  off.remove();              // 释放旧的
  off = createGraphics(w, h);
  off.noStroke();
}
