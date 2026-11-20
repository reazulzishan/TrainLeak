/* theme before first paint: ?theme=light|dark, else the system setting */
(()=>{const q=new URLSearchParams(location.search).get("theme");const t=q==="light"||q==="dark"?q:(matchMedia("(prefers-color-scheme: light)").matches?"light":"dark");document.documentElement.dataset.theme=t;})();
