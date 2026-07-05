(function () {
  "use strict";

  // --- Gradient background layer ---
  const gradientBg = document.createElement("div");
  gradientBg.id = "gradient-bg";
  document.body.prepend(gradientBg);

  // --- Canvas particle layer ---
  const canvas = document.createElement("canvas");
  canvas.id = "particle-canvas";
  document.body.prepend(canvas);

  const ctx = canvas.getContext("2d");
  let width, height;
  const PARTICLE_COUNT = 35;
  const particles = [];

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }

  function createParticle() {
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.2 - 0.1,
      radius: Math.random() * 2 + 0.5,
      baseAlpha: Math.random() * 0.4 + 0.2,
      alpha: 0,
      twinkleSpeed: Math.random() * 0.02 + 0.005,
      twinklePhase: Math.random() * Math.PI * 2,
    };
  }

  function init() {
    resize();
    particles.length = 0;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push(createParticle());
    }
  }

  function update() {
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;

      // Wrap around edges
      if (p.x < -10) p.x = width + 10;
      if (p.x > width + 10) p.x = -10;
      if (p.y < -10) p.y = height + 10;
      if (p.y > height + 10) p.y = -10;

      // Twinkle
      p.twinklePhase += p.twinkleSpeed;
      p.alpha =
        p.baseAlpha * (0.5 + 0.5 * Math.sin(p.twinklePhase));
    }
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);

    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);

      // Glow effect
      const gradient = ctx.createRadialGradient(
        p.x, p.y, 0,
        p.x, p.y, p.radius * 4
      );
      gradient.addColorStop(0, `rgba(0, 212, 255, ${p.alpha})`);
      gradient.addColorStop(0.4, `rgba(0, 212, 255, ${p.alpha * 0.4})`);
      gradient.addColorStop(1, "rgba(0, 212, 255, 0)");

      ctx.fillStyle = gradient;
      ctx.fill();

      // Core bright dot
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(200, 240, 255, ${p.alpha * 1.2})`;
      ctx.fill();
    }
  }

  function animate() {
    update();
    draw();
    requestAnimationFrame(animate);
  }

  window.addEventListener("resize", resize);
  init();
  animate();
})();
