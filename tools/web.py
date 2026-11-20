"""Local Wren-1.5 page: python tools/web.py, then open http://localhost:8000"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from cadello import Analyser

w = Analyser("wren-1.5")

PAGE = """<!doctype html><meta charset="utf-8"><title>Wren-1.5</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px 16px;background:#f5f4f0;color:#1f2328;
  font:16px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif}
main{width:100%;max-width:560px;background:#fff;border:1px solid #e4e2dc;border-radius:10px;padding:32px}
h1{margin:0 0 4px;font:600 22px/1.2 Georgia,"Times New Roman",serif;letter-spacing:.2px}
p{margin:0 0 24px;color:#6b6f76;font-size:14px}
textarea{width:100%;height:120px;resize:none;padding:12px 14px;font:inherit;color:inherit;background:#fbfbf9;
  border:1px solid #d9d7d0;border-radius:6px;outline:none;transition:border-color .15s}
textarea:focus{border-color:#1f2328;background:#fff}
.row{display:flex;justify-content:space-between;align-items:center;margin-top:12px}
#n{font-size:13px;color:#8a8e95;font-variant-numeric:tabular-nums}
button{padding:9px 20px;font-family:inherit;font-size:14px;font-weight:500;color:#fff;background:#1f2328;border:0;border-radius:6px;cursor:pointer}
button:hover{background:#3a3f45}button:disabled{opacity:.5;cursor:default}
#out{margin-top:24px;padding:16px 18px;border-left:4px solid;border-radius:6px;display:none;align-items:baseline;gap:14px}
#s{font-size:18px;letter-spacing:2px}#l{font-weight:600}
</style>
<main>
<h1>Wren-1.5</h1>
<p>Five-class sentiment analysis</p>
<textarea id="t" maxlength="200" placeholder="Enter a review"></textarea>
<div class="row"><span id="n">0 / 200</span><button id="b" onclick="go()">Classify</button></div>
<div id="out"><span id="s"></span><span id="l"></span></div>
</main>
<script>
const C = {1:"#b23a3a",2:"#c96a2b",3:"#7a7f87",4:"#3f8f5f",5:"#2b6e47"};
t.oninput = () => n.textContent = t.value.length + " / 200";
t.onkeydown = e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) go(); };
async function go(){
  if (!t.value.trim()) return;
  b.disabled = true;
  try {
    const r = await (await fetch("/predict",{method:"POST",body:t.value})).json();
    s.textContent = "★".repeat(r.stars) + "☆".repeat(5 - r.stars);
    l.textContent = r.label;
    out.style.cssText = `display:flex;color:${C[r.stars]};border-color:${C[r.stars]};background:${C[r.stars]}12`;
  } finally { b.disabled = false; }
}
</script>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(PAGE.encode())

    def do_POST(self):
        text = self.rfile.read(int(self.headers["Content-Length"])).decode()[:200] or " "
        stars = int(w.predict_proba([text])[0].argmax()) + 1
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"stars": stars, "label": w.label(stars)}).encode())


print("Wren-1.5 on http://localhost:8000")
HTTPServer(("127.0.0.1", 8000), H).serve_forever()
