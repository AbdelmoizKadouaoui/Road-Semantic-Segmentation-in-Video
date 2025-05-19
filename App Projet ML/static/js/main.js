document.addEventListener('DOMContentLoaded', () => {
  // Configuration for each upload slot
  const cfgs = [
    // Index (video only)
    { containerId: 'upload-container',       inputId: 'video-upload',         endpoint: '/upload',     field: 'video' },
    // U-Net
    { containerId: 'upload-container-video', inputId: 'video-upload-unet',    endpoint: '/unet/video', field: 'video' },
    { containerId: 'upload-container-image', inputId: 'image-upload-unet',    endpoint: '/unet/image', field: 'image' },
    // YOLO
    { containerId: 'upload-container-video', inputId: 'video-upload-yolo',    endpoint: '/yolo/video', field: 'video' },
    { containerId: 'upload-container-image', inputId: 'image-upload-yolo',    endpoint: '/yolo/image', field: 'image' },
  ];

  cfgs.forEach(cfg => {
    const container = document.getElementById(cfg.containerId);
    const fileInput = document.getElementById(cfg.inputId);
    if (!container || !fileInput) return; // skip if not on this page

    const area       = container.querySelector('.upload-area');
    const fileInfo   = container.querySelector('.file-info');
    const proc       = container.querySelector('.processing-container');
    const progress   = proc.querySelector('.progress-bar');
    const statusMsg  = proc.querySelector('[id^="status-message"]');
    const segmentBtn = proc.querySelector('.segment-btn');
    let sessionId, pollInterval;

    // Drag & drop styling
    ['dragover','dragenter'].forEach(evt =>
      area.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation();
        area.classList.add('dragover');
      })
    );
    ['dragleave','drop'].forEach(evt =>
      area.addEventListener(evt, e => {
        e.preventDefault(); e.stopPropagation();
        area.classList.remove('dragover');
      })
    );

    // Handle dropped file
    area.addEventListener('drop', e => {
      const file = e.dataTransfer.files[0];
      if (file) startUpload(file);
    });

    // Click to browse
    area.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) startUpload(file);
    });

    // “Segment” button navigates to results
    segmentBtn.addEventListener('click', () => {
      if (sessionId) {
        window.location.href = `/results/${sessionId}`;
      }
    });

    function startUpload(file) {
      fileInfo.textContent = `Selected: ${file.name}`;
      area.style.display = 'none';
      proc.style.display = 'block';
      uploadFile(file);
    }

    function uploadFile(file) {
      const fd = new FormData();
      fd.append(cfg.field, file);

      fetch(cfg.endpoint, {
        method: 'POST',
        body: fd
      })
      .then(response => {
        const ct = response.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
          // video endpoints return JSON → poll
          return response.json().then(handleJson);
        } else {
          // image endpoints return HTML → render
          return response.text().then(html => {
            document.open();
            document.write(html);
            document.close();
          });
        }
      })
      .catch(err => alert('Upload error: ' + err.message));
    }

    function handleJson(json) {
      if (json.success && json.session_id) {
        sessionId = json.session_id;
        statusMsg.textContent = json.message;
        pollProgress();
      } else {
        alert('Error: ' + (json.error || 'Unknown'));
      }
    }

    function pollProgress() {
      pollInterval = setInterval(() => {
        fetch(`/status/${sessionId}`)
          .then(r => r.json())
          .then(d => {
            progress.style.width = d.progress + '%';
            progress.textContent = d.progress + '%';
            statusMsg.textContent = d.message;

            if (d.status === 'completed') {
              clearInterval(pollInterval);
              segmentBtn.disabled = false;
              segmentBtn.classList.add('pulse');
            }
            if (d.status === 'error') {
              clearInterval(pollInterval);
              alert('Error: ' + d.message);
            }
          });
      }, 2000);
    }
  });

  // Close error modal buttons
  document.querySelectorAll('.close-modal').forEach(btn => {
    btn.addEventListener('click', () => {
      btn.closest('.modal').classList.remove('show');
    });
  });
});
