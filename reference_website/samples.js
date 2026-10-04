const samples = [
  {
    id: 1,
    title: "Forest Monitoring",
    desc: "4-panel comparison grid showing original 10m input, bicubic baseline, 2.5m-scale SR, and differences.",
    category: "forest",
    scale: "4×",
    images: [{ src: "/assets/samples/forest_sample.png", alt: "Forest Comparison" }]
  },
  {
    id: 2,
    title: "Urban Analysis",
    desc: "4-panel comparison grid showing original 10m input, bicubic baseline, 2.5m-scale SR, and differences.",
    category: "urban",
    scale: "4×",
    images: [{ src: "/assets/samples/urban_sample.png", alt: "Urban Comparison" }]
  },
  {
    id: 3,
    title: "Agricultural Observation",
    desc: "4-panel comparison grid showing original 10m input, bicubic baseline, 2.5m-scale SR, and differences.",
    category: "agriculture",
    scale: "4×",
    images: [{ src: "/assets/samples/agriculture_sample.png", alt: "Agriculture Comparison" }]
  },
  {
    id: 4,
    title: "Water Body Monitoring",
    desc: "4-panel comparison grid showing original 10m input, bicubic baseline, 2.5m-scale SR, and differences.",
    category: "water",
    scale: "4×",
    images: [{ src: "/assets/samples/water_sample.png", alt: "Water Comparison" }]
  },
  {
    id: 5,
    title: "Genuine Inference (Urban Patch)",
    desc: "Direct side-by-side comparison of 10m RGB preview and model-inferred 2.5m-scale output.",
    category: "urban",
    scale: "4×",
    images: [
      { src: "/assets/samples/real_urban_r1024_c5120_rgb.png", alt: "10m Input", label: "10m Input" },
      { src: "/assets/samples/real_urban_r1024_c5120_sr_rgb.png", alt: "2.5m Output", label: "2.5m Output" }
    ]
  },
  {
    id: 6,
    title: "TTA Disagreement Heatmap",
    desc: "Test-Time Augmentation (TTA) disagreement proxy showing areas of internal model uncertainty.",
    category: "urban",
    scale: "4×",
    images: [{ src: "/assets/samples/tta_disagreement_proxy.png", alt: "TTA Proxy" }]
  }
];

const grid = document.getElementById('gallery-grid');
const filterBtns = document.querySelectorAll('.filter-btn');
const lightbox = document.getElementById('lightbox');
const lightboxContent = document.getElementById('lightbox-content');
const lightboxCaption = document.getElementById('lightbox-caption');
const lightboxClose = document.getElementById('lightbox-close');

function renderGallery(filter = 'all') {
  grid.innerHTML = '';
  
  const filtered = filter === 'all' 
    ? samples 
    : samples.filter(s => s.category === filter);
    
  filtered.forEach(sample => {
    const card = document.createElement('div');
    card.className = 'card';
    
    // Images Container
    const imgContainer = document.createElement('div');
    imgContainer.className = `card-images ${sample.images.length === 1 ? 'single-image' : ''}`;
    
    sample.images.forEach((img, i) => {
      const imgEl = document.createElement('img');
      imgEl.src = img.src;
      imgEl.alt = img.alt;
      imgEl.loading = 'lazy';
      
      if (img.label) {
        const label = document.createElement('div');
        label.className = `img-label ${i === 0 ? 'label-left' : 'label-right'}`;
        label.textContent = img.label;
        imgContainer.appendChild(label);
      }
      imgContainer.appendChild(imgEl);
    });
    
    imgContainer.onclick = () => openLightbox(sample);
    
    // Body
    const body = document.createElement('div');
    body.className = 'card-body';
    
    const tags = document.createElement('div');
    tags.className = 'card-tags';
    tags.innerHTML = `
      <span class="tag">${sample.category}</span>
      <span class="tag tag-scale">${sample.scale}</span>
    `;
    
    const title = document.createElement('h3');
    title.className = 'card-title';
    title.textContent = sample.title;
    
    const desc = document.createElement('p');
    desc.className = 'card-desc';
    desc.textContent = sample.desc;
    
    body.appendChild(tags);
    body.appendChild(title);
    body.appendChild(desc);
    
    card.appendChild(imgContainer);
    card.appendChild(body);
    
    grid.appendChild(card);
  });
}

function openLightbox(sample) {
  lightboxContent.innerHTML = '';
  
  sample.images.forEach(img => {
    const imgEl = document.createElement('img');
    imgEl.src = img.src;
    imgEl.alt = img.alt;
    lightboxContent.appendChild(imgEl);
  });
  
  lightboxCaption.textContent = sample.title;
  lightbox.classList.add('active');
  document.body.style.overflow = 'hidden';
}

lightboxClose.onclick = () => {
  lightbox.classList.remove('active');
  document.body.style.overflow = '';
};

lightbox.onclick = (e) => {
  if (e.target === lightbox) {
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
  }
};

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && lightbox.classList.contains('active')) {
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
  }
});

filterBtns.forEach(btn => {
  btn.onclick = () => {
    filterBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderGallery(btn.dataset.filter);
  };
});

// Init
document.addEventListener('DOMContentLoaded', () => {
  renderGallery('all');
});
