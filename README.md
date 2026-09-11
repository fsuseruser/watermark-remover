---
title: Watermark Remover
emoji: 🧼
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "1.63.0"
app_file: app.py
pinned: false
---

# Watermark Remover (Streamlit + LaMa)

A Streamlit app that removes watermarks using **LaMa**, a real pretrained
neural inpainting model — the same category of technology behind tools like
Photoshop's Generative Fill. Unlike a hand-written blur/diffusion filter,
it reconstructs plausible detail behind the mark (door edges, tile lines,
cabinet fronts) instead of smearing colors together.

## Install

```bash
pip install -r requirements.txt
```

The first time you actually run a removal, it downloads the model weights
(~200MB) from GitHub to `~/.cache/torch/hub/checkpoints/big-lama.pt` and
caches them there for all future runs.

**If `pip install torch` pulls in several GB of CUDA packages** (common on
Linux even when you only want CPU inference) and you don't have a GPU,
either:
- Install the official CPU-only build first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then run `pip install -r requirements.txt` again, or
- Install torch with `--no-deps` (works because this app never touches CUDA): `pip install torch --no-deps`, then install the rest of `requirements.txt` normally.

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Using it

1. **Upload photos** in the sidebar (multiple at once is fine).
2. Choose how the watermark is positioned:
   - **Same spot on every photo** — for a batch that shares one fixed
     watermark position (e.g. an agency logo stamped at the same pixel
     location across an export). Set the rectangle once on a reference
     photo; it's applied proportionally to every other photo.
   - **Different spot per photo** — set a rectangle individually per photo
     in its own tab.
3. Adjust the **Left / Top / Width / Height** number inputs until the red
   preview box fully covers the watermark.
4. Click **Remove watermarks**. The first click loads the model (a few
   seconds); after that each photo takes a few seconds on CPU.
5. Review results (toggle "Show original" per photo to compare), then
   **Download all as .zip**.

## Honest limitations

- Works best when the masked rectangle covers the watermark with a little
  margin, but not much more — an overly large box removes real photo
  content the model then has to invent from nothing.
- Fine repetitive texture (e.g. distant glass curtain-wall windows) is the
  model's weakest case — it comes out plausible but slightly soft rather
  than pixel-perfect.
- This is a rectangle-mask tool, not a segmentation tool — irregularly
  shaped watermarks still need a rectangle that fully contains them.
- CPU inference is fine for occasional use but slow for large batches
  (roughly a few seconds per photo depending on size and hardware). A GPU
  machine with a CUDA build of torch will be much faster if you're doing
  this at volume.

## Deploying this online

This app is heavier than a typical Streamlit demo — it bundles PyTorch and
loads a real neural network into memory — so the host you pick matters more
than usual.

**Hugging Face Spaces no longer has a free option for this.** As of 2026,
Spaces only offers free hosting for *Static* sites (no Python). Running
actual code (Gradio or Docker, which is how Streamlit apps deploy there
now) requires a paid PRO plan ($9/month). If you already pay for PRO, it's
a great fit — 16GB RAM comfortably handles PyTorch + LaMa, and it reads
this repo's `requirements.txt` and `packages.txt` automatically. Steps:
1. Create a Space, SDK = Docker (or Gradio).
2. Push these files (`app.py`, `lama_inpaint.py`, `requirements.txt`,
   `packages.txt`, `.streamlit/config.toml`) to the Space's repo.
3. It builds and serves automatically at `<your-space>.hf.space`.

**Streamlit Community Cloud is the actual free option.** Its free tier is
memory-tight (roughly 1GB) — PyTorch plus a few in-flight images can get
close to that ceiling, especially with more than one concurrent user — but
it costs nothing. Steps:
1. Push this repo to GitHub.
2. Go to share.streamlit.io, sign in with GitHub, click **New app**, and
   point it at your repo and `app.py`.
3. It reads `requirements.txt` and `packages.txt` automatically.

Community Cloud has a known, currently-unresolved bug where it silently
ignores any Python-version pin (a `runtime.txt` file or the "Advanced
settings" dropdown) and just uses whatever version it feels like — which
has been as new as 3.14 for some deployments. That's exactly why this repo
vendors its own copy of the LaMa code instead of depending on a PyPI
package with old pinned versions (see below) — it means the app doesn't
care which Python version you end up running on.

**A plain VM/container (Fly.io, Render, Railway, your own server)** gives
you the most headroom and is worth it if you expect real usage — pick at
least 2GB RAM.

### Files in this repo that matter for deployment
- `requirements.txt` — pulls CPU-only torch from PyTorch's own index
  (`--extra-index-url`) instead of the default PyPI build, which otherwise
  drags in several GB of unneeded CUDA packages and can blow build-time
  disk quotas on hosted platforms.
- `packages.txt` — installs `libgl1` and `libglib2.0-0`, which opencv
  needs to even *import* on minimal Debian-based images. Without this the
  app crashes on startup with a `libGL.so.1: cannot open shared object
  file` error — an easy one to lose an hour to.
- `.streamlit/config.toml` — caps uploads at 20MB/file (raise if your
  photos are bigger) and sets `headless = true` for server environments.
- `lama_inpaint.py` — a vendored copy of the LaMa inference class (see
  below) instead of installing `simple-lama-inpainting` from PyPI.

**Why vendor instead of `pip install simple-lama-inpainting`:** that
package declares dependencies on old pinned versions of Pillow/numpy it
doesn't actually need at runtime (only its unused CLI wrapper needs them).
On a host that ends up running a newer Python than those old pins have
prebuilt wheels for, pip falls back to building Pillow from source — which
fails outright on modern setuptools. This bit us directly on Streamlit
Community Cloud. Vendoring the ~150 lines of actual inference code
sidesteps the problem entirely: the app runs on whatever modern
Pillow/numpy/opencv the host already has, regardless of Python version.


### Things worth knowing before you get real traffic
- **Cold starts re-download the model** on platforms with ephemeral
  storage (the ~200MB LaMa weights aren't bundled in the repo, so a fresh
  container fetches them from GitHub on first use). Fine for Spaces/most
  PaaS since the disk usually persists across normal restarts; if your
  host truly wipes storage every boot, consider baking the weights into
  your image instead.
- **CPU inference is shared across all visitors** — Streamlit runs one
  process per app instance, so if several people click "Remove" at once,
  they queue behind each other. Fine for casual/internal use; for real
  public traffic you'd want a GPU instance or a task queue in front of it.
- **Uploaded photos stay in memory only** — this app never writes uploads
  to disk, so there's nothing to clean up between sessions. Still worth
  saying explicitly in your own privacy notice if photos might be
  sensitive (e.g. real estate listings with addresses visible).
- Consider capping the number of files per batch and/or adding basic auth
  if this goes fully public, since each image costs real CPU time and
  there's no built-in rate limiting.
