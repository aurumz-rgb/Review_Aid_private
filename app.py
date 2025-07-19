import streamlit as st
import streamlit.components.v1 as components

# Minimal particles config JSON as a string
particles_config = """
{
  "autoPlay": true,
  "background": { "color": { "value": "#243660" } },
  "fullScreen": { "enable": true, "zIndex": 0 },
  "fpsLimit": 120,
  "interactivity": {
    "detectsOn": "window",
    "events": {
      "onClick": { "enable": true, "mode": "push" },
      "onHover": { "enable": true, "mode": "repulse" },
      "resize": { "enable": true }
    },
    "modes": {
      "push": { "quantity": 4 },
      "repulse": { "distance": 200, "duration": 0.4 }
    }
  },
  "particles": {
    "color": { "value": "#FF9B45" },
    "links": {
      "enable": true,
      "color": "#ffffff",
      "distance": 150,
      "opacity": 0.2,
      "width": 1
    },
    "move": { "enable": true, "speed": 2, "outModes": { "default": "out" } },
    "number": { "value": 80, "density": { "enable": true, "area": 800 } },
    "opacity": {
      "value": { "min": 0.1, "max": 0.5 },
      "animation": { "enable": true, "speed": 3, "startValue": "random" }
    },
    "shape": { "type": "circle" },
    "size": {
      "value": { "min": 0.1, "max": 5 },
      "animation": { "enable": true, "speed": 10, "startValue": "random" }
    },
    "twinkle": {
      "lines": { "enable": true, "frequency": 0.0002, "opacity": 1, "color": "#E6521F" },
      "particles": { "enable": true, "frequency": 0.05, "opacity": 1, "color": "#ffff00" }
    }
  }
}
"""

# Full HTML for particles.js (using CDN)
particles_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Particles</title>
  <style>
    /* Make canvas fill the entire window */
    body, html {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background-color: #0d47a1;
    }}
    #tsparticles {{
      position: fixed;
      width: 100%;
      height: 100%;
      top: 0; left: 0;
      z-index: 0;
    }}
  </style>
</head>
<body>
  <div id="tsparticles"></div>

  <!-- Load particles.js library from CDN -->
  <script src="https://cdn.jsdelivr.net/npm/tsparticles@2/tsparticles.bundle.min.js"></script>

  <script>
    window.onload = function() {{
      tsParticles.load("tsparticles", {particles_config});
    }};
  </script>
</body>
</html>
"""

st.set_page_config(layout="wide", page_title="Particles Background")

# Embed the HTML component and fill the entire screen height
components.html(particles_html, height=1000, scrolling=False)
