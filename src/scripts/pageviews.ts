const BUSUANZI_SRC =
  "https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js";
const SCRIPT_ID = "busuanzi-script";

function loadBusuanzi() {
  document.getElementById(SCRIPT_ID)?.remove();

  const script = document.createElement("script");
  script.id = SCRIPT_ID;
  script.async = true;
  script.src = BUSUANZI_SRC;
  document.body.appendChild(script);
}

loadBusuanzi();
document.addEventListener("astro:page-load", loadBusuanzi);
