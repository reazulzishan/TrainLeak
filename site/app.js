// TrainLeak web: runs the real Python package in the browser with Pyodide. Files are read locally and never sent anywhere.
"use strict";

const $ = (id) => document.getElementById(id);
const DATA = ".csv,.tsv,.txt,.parquet,.json";
const CODE = ".ipynb,.py";
let py = null;
let mode = "split";
let last = null; // {md, json, name}

function status(msg, pct, err) {
  $("status").textContent = msg;
  $("status").className = err ? "err" : "";
  $("live").textContent = msg;
  if (pct !== undefined) $("bar").style.width = pct + "%";
}

async function boot() {
  try {
    status("Loading Python in your browser… (first visit downloads about 30 MB, then it is cached)", 10);
    py = await loadPyodide();
    status("Loading pandas and scikit-learn…", 40);
    await py.loadPackage(["pandas", "numpy", "scikit-learn", "micropip"]);
    status("Installing TrainLeak…", 80);
    const wheel = (await (await fetch("wheel.txt", { cache: "no-store" })).text()).trim();
    await py.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(new URL(wheel, location.href).href)}, deps=False)
import json, trainleak
from trainleak.data import _frame

def _run(kind, a=None, b=None, target=None):
    if kind == "demo":
        r = trainleak.demo()
    elif kind == "code":
        r = trainleak.check_code(a)
    else:
        r = trainleak.check(a, b, target=target or None)
    return json.dumps({"html": r._repr_html_(), "md": r.markdown(), "json": r.json(), "ok": r.ok, "verdict": r.verdict})

def _columns(path):
    return json.dumps([str(c) for c in _frame(path, "x").columns])
`);
    const v = py.runPython("trainleak.__version__");
    status(`Ready · TrainLeak ${v}. Drop your files, or try an example below.`, 100);
    setTimeout(() => { $("barwrap").hidden = true; }, 600);
    refresh();
  } catch (e) {
    console.error(e);
    status("Could not start Python in this browser. Try a recent Chrome, Edge, Firefox or Safari.", 100, true);
  }
}

function setMode(m) {
  mode = m;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
  const two = m === "split";
  $("drops").className = "drops" + (two ? " two" : "");
  $("d2").hidden = !two;
  $("opts").hidden = m === "code";
  $("t1").textContent = m === "split" ? "Training file" : m === "one" ? "Your dataset (before splitting)" : "Your notebook or script";
  $("s1").textContent = m === "code" ? "A .ipynb or .py file: read, never run" : "CSV, TSV, Parquet or JSON";
  $("f1").accept = m === "code" ? CODE : DATA;
  $("f2").accept = DATA;
  for (const n of ["f1", "f2"]) { $(n).value = ""; $(n === "f1" ? "d1" : "d2").classList.remove("has"); }
  $("t2").textContent = "Test file";
  resetTarget();
  refresh();
}

function resetTarget() {
  $("target").innerHTML = '<option value="">guess automatically</option>';
  $("target").disabled = true;
}

function refresh() {
  const need = mode === "split" ? [$("f1"), $("f2")] : [$("f1")];
  $("run").disabled = !py || !need.every((f) => f.files.length);
}

async function toFS(file) {
  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf("."));
  if (ext === ".parquet" && !py.loadedPackages.pyarrow) { status("Loading Parquet support…"); await py.loadPackage("pyarrow"); }
  if ([".xlsx", ".xls"].includes(ext)) throw new Error("Excel files aren't supported in the browser yet. Save the sheet as CSV and try again.");
  const safe = file.name.replace(/[^\w.\-]+/g, "_").slice(-120) || "data" + ext;
  const dir = "/tmp/u" + Date.now() + Math.floor(Math.random() * 1e6);
  py.FS.mkdirTree(dir);
  const path = dir + "/" + safe;
  py.FS.writeFile(path, new Uint8Array(await file.arrayBuffer()));
  return path;
}

async function onFile(input, box, label) {
  if (!input.files.length) return;
  const f = input.files[0];
  $(box).classList.add("has");
  $(label).textContent = f.name;
  refresh();
  if (mode !== "code" && input.id === "f1" && py) {
    try {
      const cols = JSON.parse(py.globals.get("_columns")(await toFS(f)));
      $("target").innerHTML = '<option value="">guess automatically</option>' + cols.map((c) => `<option>${escapeHtml(c)}</option>`).join("");
      $("target").disabled = false;
    } catch (e) {
      resetTarget();
      status("Couldn't read the columns of " + f.name + ": " + friendly(e), undefined, true);
      return;
    }
    status("Ready. Pick the target column (or let TrainLeak guess), then press Check for leaks.");
  }
}

function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

function friendly(e) {
  const msg = String(e && e.message || e);
  const line = msg.trim().split("\n").filter(Boolean).pop() || msg;
  return line.replace(/^\w+Error: /, "");
}

async function run(kind, a, b, name) {
  $("run").disabled = true;
  status("Checking…", 100);
  $("result").innerHTML = "";
  $("dl").hidden = true;
  await new Promise((r) => setTimeout(r, 30)); // let the status paint
  try {
    const t = performance.now();
    const out = JSON.parse(py.globals.get("_run")(kind, a ?? null, b ?? null, $("target").value || null));
    $("result").innerHTML = out.html; // built and escaped by trainleak.Report._repr_html_
    last = { md: out.md, json: out.json, name };
    $("dl").hidden = false;
    status(`Done in ${((performance.now() - t) / 1000).toFixed(1)} s · ${out.verdict}`);
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    console.error(e);
    status("Couldn't check this: " + friendly(e), undefined, true);
  } finally { refresh(); }
}

function download(text, filename, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

async function example(kind) {
  if (!py) return;
  if (kind === "demo") return run("demo", null, null, "demo");
  const file = kind === "nb" ? "smote_mistake.ipynb" : "smote_fixed.ipynb";
  const bytes = new Uint8Array(await (await fetch("examples/" + file)).arrayBuffer());
  py.FS.writeFile("/tmp/" + file, bytes);
  run("code", "/tmp/" + file, null, file);
}

document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
$("f1").addEventListener("change", () => onFile($("f1"), "d1", "t1"));
$("f2").addEventListener("change", () => onFile($("f2"), "d2", "t2"));
for (const id of ["d1", "d2"]) {
  const d = $(id);
  d.addEventListener("dragover", () => d.classList.add("over"));
  d.addEventListener("dragleave", () => d.classList.remove("over"));
  d.addEventListener("drop", () => d.classList.remove("over"));
}
$("run").addEventListener("click", async () => {
  try {
    if (mode === "code") return run("code", await toFS($("f1").files[0]), null, $("f1").files[0].name);
    const a = await toFS($("f1").files[0]);
    const b = mode === "split" ? await toFS($("f2").files[0]) : null;
    run("data", a, b, $("f1").files[0].name);
  } catch (e) { status("Couldn't read the file: " + friendly(e), undefined, true); }
});
$("ex-demo").addEventListener("click", () => example("demo"));
$("ex-nb").addEventListener("click", () => example("nb"));
$("ex-fixed").addEventListener("click", () => example("fixed"));
$("md").addEventListener("click", () => last && download(last.md, "trainleak_report.md", "text/markdown"));
$("js").addEventListener("click", () => last && download(last.json, "trainleak_report.json", "application/json"));
setMode("split");
boot();
