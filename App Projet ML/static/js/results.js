document.addEventListener('DOMContentLoaded', () => {
  // Collect either video or image elements for each mode
  const items = {
    original:  document.getElementById('original-video')  || document.getElementById('original-img'),
    segmented: document.getElementById('segmented-video')|| document.getElementById('segmented-img'),
    grayscale: document.getElementById('grayscale-video')|| document.getElementById('grayscale-img'),
    overlay:   document.getElementById('overlay-video')  || document.getElementById('overlay-img'),
  };

  // Label element for videos or images
  const label   = document.getElementById('video-label') || document.getElementById('image-label');
  const buttons = document.querySelectorAll('.toggle-btn');
  const dlBtn   = document.getElementById('download-btn');

  // Mode‐switching buttons
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;

      // Highlight active button
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Show the chosen element, hide others
      Object.entries(items).forEach(([key, el]) => {
        if (!el) return;
        if (key === mode) {
          el.classList.remove('hidden');
          // If it's a video, reset and play
          if (el.tagName === 'VIDEO') {
            el.currentTime = 0;
            el.load();
            el.play().catch(() => {});
          }
          if (label) label.textContent = btn.textContent.trim();
        } else {
          el.classList.add('hidden');
          if (el.tagName === 'VIDEO') el.pause();
        }
      });
    });
  });

  // Smooth scroll to download panel
  if (dlBtn) {
    dlBtn.addEventListener('click', () => {
      document
        .querySelector('.download-panel')
        .scrollIntoView({ behavior: 'smooth' });
    });
  }

  // Debug logging (optional)
  Object.values(items).forEach(el => {
    if (!el) return;
    el.addEventListener('loadedmetadata', () => {
      console.log(
        `[MEDIA] ${el.id} loaded: duration=${el.duration || el.naturalWidth} ${
          el.tagName
        }`
      );
    });
    el.addEventListener('error', () => {
      console.error(
        `[MEDIA] ${el.id} ERROR: code=${el.error?.code}, message=${
          el.error?.message || 'unknown'
        }`
      );
    });
  });
});
