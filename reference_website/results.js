document.addEventListener('DOMContentLoaded', () => {
  const rawData = sessionStorage.getItem('inferenceResult');
  
  if (!rawData) {
    document.getElementById('error-state').style.display = 'block';
    return;
  }

  const data = JSON.parse(rawData);
  const API_URL = 'http://127.0.0.1:8000';

  // Show the content block
  document.getElementById('results-content').style.display = 'flex';

  // 1. Comparison Images
  if (data.artifacts) {
    if (data.artifacts.lr_rgb_preview) {
      document.getElementById('img-lr').src = `${API_URL}${data.artifacts.lr_rgb_preview}`;
    }
    if (data.artifacts.rgb_preview) {
      document.getElementById('img-sr').src = `${API_URL}${data.artifacts.rgb_preview}`;
    }
    
    // Downloads
    if (data.artifacts.geotiff) document.getElementById('dl-geotiff').href = `${API_URL}${data.artifacts.geotiff}`;
    if (data.artifacts.reflectance_npy) document.getElementById('dl-refl').href = `${API_URL}${data.artifacts.reflectance_npy}`;
    if (data.artifacts.uint16_npy) document.getElementById('dl-uint16').href = `${API_URL}${data.artifacts.uint16_npy}`;
  }

  // 2. Metrics
  if (data.quality_metrics) {
    document.getElementById('val-tta').textContent = data.quality_metrics.mean_tta_disagreement?.toFixed(4) || "-";
    document.getElementById('val-rad').textContent = data.quality_metrics.mean_radiometric_consistency_residual?.toFixed(4) || "-";
    document.getElementById('val-ndvi').textContent = data.quality_metrics.mean_absolute_ndvi_consistency?.toFixed(4) || "-";
  }
  
  document.getElementById('val-clip').textContent = data.clipped_uint16_pct?.toFixed(2) || "0.00";
  document.getElementById('val-time').textContent = data.inference_time_sec?.toFixed(2) || "-";

  // 3. Quality Maps
  if (data.quality_maps) {
    if (data.quality_maps.radiometric_consistency_residual) {
      document.getElementById('map-rad').src = `${API_URL}${data.quality_maps.radiometric_consistency_residual}`;
    }
    if (data.quality_maps.sr_vs_bicubic_difference) {
      document.getElementById('map-diff').src = `${API_URL}${data.quality_maps.sr_vs_bicubic_difference}`;
    }
    if (data.quality_maps.tta_disagreement_proxy) {
      document.getElementById('map-tta').src = `${API_URL}${data.quality_maps.tta_disagreement_proxy}`;
    }
    if (data.quality_maps.ndvi_consistency_residual) {
      document.getElementById('map-ndvi-10').src = `${API_URL}${data.quality_maps.ndvi_consistency_residual}`;
    }
    if (data.quality_maps.sr_ndvi_2p5m) {
      document.getElementById('map-ndvi-25').src = `${API_URL}${data.quality_maps.sr_ndvi_2p5m}`;
    }
    if (data.quality_maps.ndvi_consistency_residual_display) {
      document.getElementById('map-ndvi-disp').src = `${API_URL}${data.quality_maps.ndvi_consistency_residual_display}`;
    }
  }
  
  // Force page to start at the top
  window.scrollTo(0, 0);
});
