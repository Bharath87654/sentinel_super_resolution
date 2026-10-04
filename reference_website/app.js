const menuToggle = document.querySelector('.menu-toggle');
const primaryNav = document.querySelector('#primary-nav');
const hero = document.querySelector('.hero');

const fileInput       = document.querySelector('#image-input');
const sampleButton    = document.querySelector('#use-sample');
const previewWrap     = document.querySelector('#local-preview');
const previewImage    = document.querySelector('#local-preview-image');
const previewName     = document.querySelector('#local-preview-name');
const fileStatus      = document.querySelector('#file-status');

const generateButton  = document.querySelector('#generate-button');
const modelStatus     = document.querySelector('#model-status');

// GeoTIFF Validation Card (reuses the same IDs from index.html)
const metaBands  = document.querySelector('#meta-bands');
const metaShape  = document.querySelector('#meta-shape');   // repurposed: pixel size
const metaCrs    = document.querySelector('#meta-crs');     // repurposed: dimensions
const metaPixel  = document.querySelector('#meta-pixel');   // repurposed: file size
const metaStatus = document.querySelector('#meta-status');

const downloadLinksContainer = document.querySelector('#download-links');
const downloadButton         = document.querySelector('#download-button');
const downloadStatus         = document.querySelector('#download-status');

// Compact Results Elements
const compactView     = document.querySelector('#results-compact-view');
const placeholderView = document.querySelector('#results-placeholder-view');

const API_URL = 'http://127.0.0.1:8000';

// ── State ─────────────────────────────────────────────────────────────────────
let currentGeotiff = null;   // File | null

// ── Mobile menu ───────────────────────────────────────────────────────────────
if (menuToggle && primaryNav) {
  const closeMenu = () => {
    menuToggle.setAttribute('aria-expanded', 'false');
    primaryNav.classList.remove('is-open');
  };
  menuToggle.addEventListener('click', () => {
    const isOpen = menuToggle.getAttribute('aria-expanded') === 'true';
    menuToggle.setAttribute('aria-expanded', String(!isOpen));
    primaryNav.classList.toggle('is-open', !isOpen);
  });
  primaryNav.querySelectorAll('a').forEach((link) => link.addEventListener('click', closeMenu));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeMenu();
  });
}

// ── Hero parallax ─────────────────────────────────────────────────────────────
if (hero && window.matchMedia('(pointer: fine)').matches && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  let framePending = false;
  let latestEvent;
  hero.addEventListener('pointermove', (event) => {
    latestEvent = event;
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(() => {
      const rect = hero.getBoundingClientRect();
      const x = Math.max(-1, Math.min(1, (latestEvent.clientX - rect.left) / rect.width * 2 - 1));
      const y = Math.max(-1, Math.min(1, (latestEvent.clientY - rect.top) / rect.height * 2 - 1));
      hero.style.setProperty('--parallax-x', `${-x * 7}px`);
      hero.style.setProperty('--parallax-y', `${-y * 5}px`);
      hero.style.setProperty('--tilt-x', `${-y * 3}deg`);
      hero.style.setProperty('--tilt-y', `${x * 4}deg`);
      hero.style.setProperty('--card-x', `${x * 3}px`);
      hero.style.setProperty('--card-y', `${y * 2}px`);
      framePending = false;
    });
  });
  hero.addEventListener('pointerleave', () => {
    ['--parallax-x', '--parallax-y', '--tilt-x', '--tilt-y', '--card-x', '--card-y'].forEach(
      (p) => hero.style.removeProperty(p)
    );
  });
}

// ── Reset UI ──────────────────────────────────────────────────────────────────
const clearFiles = (keepInput = false) => {
  currentGeotiff = null;
  if (!keepInput && fileInput) fileInput.value = '';

  if (previewImage)  previewImage.src = '';
  if (previewName)   previewName.textContent = '';
  if (previewWrap)   previewWrap.hidden = true;

  if (fileStatus)  fileStatus.textContent  = 'No file selected. Upload a 4-band Sentinel-2 GeoTIFF (.tif / .tiff).';
  if (modelStatus) modelStatus.textContent = 'Ready to run.';

  if (generateButton) generateButton.disabled = true;

  if (metaBands)  metaBands.textContent  = '-';
  if (metaShape)  metaShape.textContent  = '-';
  if (metaCrs)    metaCrs.textContent    = '-';
  if (metaPixel)  metaPixel.textContent  = '-';
  if (metaStatus) metaStatus.textContent = 'Waiting for GeoTIFF upload.';

  if (downloadLinksContainer) downloadLinksContainer.style.display = 'none';
  if (downloadStatus)         downloadStatus.textContent = 'Awaiting completion.';

  if (compactView)     compactView.style.display     = 'none';
  if (placeholderView) placeholderView.style.display = 'block';
};

if (sampleButton) {
  sampleButton.addEventListener('click', () => clearFiles(false));
}

if (downloadButton) {
  downloadButton.addEventListener('click', () => {
    const samplesSection = document.querySelector('#samples');
    if (samplesSection) samplesSection.scrollIntoView({ behavior: 'smooth' });
  });
}

// ── File selection: client-side validation ────────────────────────────────────
if (fileInput) {
  if (generateButton) generateButton.disabled = true;

  fileInput.addEventListener('change', () => {
    clearFiles(true);

    const files = fileInput.files;
    if (!files || files.length === 0) return;

    if (files.length > 1) {
      if (fileStatus) fileStatus.textContent =
        'Error: Please select exactly one GeoTIFF file. Multiple files are not supported.';
      return;
    }

    const file = files[0];
    const nameLower = file.name.toLowerCase();

    // Extension guard — before anything else
    if (!nameLower.endsWith('.tif') && !nameLower.endsWith('.tiff')) {
      if (fileStatus) fileStatus.textContent =
        `Error: "${file.name}" is not a GeoTIFF. Only .tif and .tiff files are accepted. ` +
        'JPG, PNG, and NPY files are not supported.';
      return;
    }

    currentGeotiff = file;

    // Show what the browser can read locally (server does the real validation)
    const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);

    if (metaBands)  metaBands.textContent  = 'B02, B03, B04, B08 (verified by server)';
    if (metaShape)  metaShape.textContent  = '~10 m (verified by server)';
    if (metaCrs)    metaCrs.textContent    = 'Read from file by server';
    if (metaPixel)  metaPixel.textContent  = `${fileSizeMB} MB`;
    if (metaStatus) metaStatus.textContent = 'File accepted locally. Server will verify band order and pixel size on upload.';

    if (previewName)   previewName.textContent = file.name;
    if (previewWrap)   previewWrap.hidden = false;

    if (fileStatus) fileStatus.textContent =
      `"${file.name}" selected (${fileSizeMB} MB). Centre 64\u00d764 patch will be extracted.`;

    if (generateButton) generateButton.disabled = false;
  });
}

// ── Inference ─────────────────────────────────────────────────────────────────
if (generateButton && modelStatus) {
  generateButton.addEventListener('click', async () => {
    if (!currentGeotiff) return;

    generateButton.disabled = true;
    generateButton.textContent = 'Running Inference\u2026';
    modelStatus.textContent = 'Uploading and validating GeoTIFF\u2026';

    const formData = new FormData();
    formData.append('geotiff', currentGeotiff);

    try {
      const response = await fetch(`${API_URL}/infer-geotiff`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok || data.status !== 'success') {
        // Surface the server's full, human-readable validation message
        throw new Error(data.detail || data.message || 'API returned an error.');
      }

      // Update metadata card with server-confirmed values
      if (metaBands)  metaBands.textContent  = 'B02, B03, B04, B08 \u2713';
      if (metaShape)  metaShape.textContent  = `${data.input_gsd_m} m \u2713`;
      if (metaCrs)    metaCrs.textContent    = data.crs || '\u2014';
      if (metaPixel)  metaPixel.textContent  = (currentGeotiff.size / (1024 * 1024)).toFixed(2) + ' MB';
      if (metaStatus) metaStatus.textContent = 'All server validations passed \u2713';

      modelStatus.textContent =
        `Inference complete in ${data.inference_time_sec?.toFixed(2)}s. Centre 64\u00d764 patch processed.`;

      // Download links
      if (downloadLinksContainer) {
        const geotiffLink = document.querySelector('#dl-geotiff');
        const reflLink    = document.querySelector('#dl-refl');
        const uint16Link  = document.querySelector('#dl-uint16');
        if (geotiffLink) geotiffLink.href = `${API_URL}${data.artifacts.geotiff}`;
        if (reflLink)    reflLink.href    = `${API_URL}${data.artifacts.reflectance_npy}`;
        if (uint16Link)  uint16Link.href  = `${API_URL}${data.artifacts.uint16_npy}`;
        downloadLinksContainer.style.display = 'flex';
        if (downloadStatus) downloadStatus.textContent = 'Processing successful! See viewer.';
      }

      // Persist for the dedicated results page (identical schema)
      sessionStorage.setItem('inferenceResult', JSON.stringify({
        artifacts:          data.artifacts,
        quality_metrics:    data.quality_metrics,
        quality_maps:       data.quality_maps,
        inference_time_sec: data.inference_time_sec,
        clipped_uint16_pct: data.clipped_uint16_pct
      }));

      if (compactView)     compactView.style.display     = 'block';
      if (placeholderView) placeholderView.style.display = 'none';

      const samplesSection = document.querySelector('#samples');
      if (samplesSection) samplesSection.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
      // Show the full server error message — it is intentionally human-readable
      modelStatus.textContent = `Error: ${err.message}`;
      if (metaStatus) metaStatus.textContent = 'Validation failed. See error message above.';
    } finally {
      generateButton.disabled = false;
      generateButton.innerHTML = `Run Inference <span aria-hidden="true">\u2192</span>`;
    }
  });
}
