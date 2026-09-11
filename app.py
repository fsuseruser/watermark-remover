import io
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="Watermark Remover (AI Inpainting)", layout="wide")

PREVIEW_MAX_WIDTH = 700


@st.cache_resource(show_spinner=False)
def load_model():
    """Loads the LaMa neural inpainting model once per session and caches it."""
    from lama_inpaint import SimpleLama
    return SimpleLama()


def build_rect_mask(img_size, rect):
    """Builds a full-resolution binary mask (PIL 'L' image) from a rectangle
    given in the image's own pixel coordinates."""
    mask = Image.new("L", img_size, 0)
    x, y, w, h = rect
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + w, y + h], fill=255)
    return mask


def preview_with_rect(image: Image.Image, rect):
    """Returns a resized preview image with the current mask rectangle
    drawn on top, so the person can see exactly what will be removed."""
    display = image.copy()
    display.thumbnail((PREVIEW_MAX_WIDTH, PREVIEW_MAX_WIDTH * 4))
    scale = display.width / image.width
    x, y, w, h = rect
    draw = ImageDraw.Draw(display, "RGBA")
    draw.rectangle(
        [x * scale, y * scale, (x + w) * scale, (y + h) * scale],
        fill=(255, 60, 60, 90),
        outline=(255, 60, 60, 255),
        width=2,
    )
    return display


def run_inpaint(model, image: Image.Image, mask: Image.Image) -> Image.Image:
    return model(image.convert("RGB"), mask.convert("L"))


def images_to_zip(named_images: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, img in named_images.items():
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="WEBP", quality=92)
            zf.writestr(name, img_bytes.getvalue())
    buf.seek(0)
    return buf.read()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

st.sidebar.title("Watermark Remover")
st.sidebar.caption(
    "Powered by LaMa neural inpainting — reconstructs plausible detail "
    "behind the mark (doors, tiles, cabinets) instead of just blurring."
)

uploaded_files = st.sidebar.file_uploader(
    "Upload photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

mode = st.sidebar.radio(
    "Watermark position",
    ["Same spot on every photo", "Different spot per photo"],
    help=(
        "Pick 'same spot' when a batch shares one fixed watermark position "
        "(e.g. a logo stamped at the same pixel location across an export). "
        "Pick 'different spot' to set a rectangle on each photo individually."
    ),
)

if not uploaded_files:
    st.title("Watermark Remover")
    st.write(
        "Upload one or more photos in the sidebar to get started. "
        "This tool uses a real pretrained inpainting model (LaMa), so it "
        "reconstructs plausible detail behind the watermark rather than "
        "just blurring colors together."
    )
    st.stop()

images = {f.name: Image.open(f).convert("RGB") for f in uploaded_files}
names = list(images.keys())


def rect_inputs(key_prefix, img_size, defaults):
    w_img, h_img = img_size
    c1, c2, c3, c4 = st.columns(4)
    x = c1.number_input("Left (x)", 0, w_img, defaults[0], key=f"{key_prefix}_x")
    y = c2.number_input("Top (y)", 0, h_img, defaults[1], key=f"{key_prefix}_y")
    w = c3.number_input("Width", 1, w_img - x, min(defaults[2], w_img - x), key=f"{key_prefix}_w")
    h = c4.number_input("Height", 1, h_img - y, min(defaults[3], h_img - y), key=f"{key_prefix}_h")
    return (x, y, w, h)


def default_rect(img_size):
    w, h = img_size
    return (int(w * 0.35), int(h * 0.4), int(w * 0.3), int(h * 0.2))


# ----------------------------------------------------------------------------
# Mask collection
# ----------------------------------------------------------------------------

masks = {}

if mode == "Same spot on every photo":
    st.subheader("1. Set the mask once")
    st.caption(
        "Pick any photo where the watermark is clearly visible, then adjust "
        "the box until it fully covers the mark. The same rectangle "
        "(scaled proportionally) is applied to every uploaded photo."
    )
    reference_name = st.selectbox("Reference photo", names)
    reference_image = images[reference_name]

    rect = rect_inputs("shared", reference_image.size, default_rect(reference_image.size))
    st.image(preview_with_rect(reference_image, rect), caption="Preview — red box will be removed", width="stretch")

    rx, ry, rw, rh = rect
    fx, fy = rx / reference_image.width, ry / reference_image.height
    fw, fh = rw / reference_image.width, rh / reference_image.height
    for name in names:
        w, h = images[name].size
        scaled_rect = (int(fx * w), int(fy * h), int(fw * w), int(fh * h))
        masks[name] = build_rect_mask((w, h), scaled_rect)

else:
    st.subheader("1. Set a mask on each photo")
    tabs = st.tabs(names)
    for name, tab in zip(names, tabs):
        with tab:
            img = images[name]
            rect = rect_inputs(name, img.size, default_rect(img.size))
            st.image(preview_with_rect(img, rect), caption="Preview — red box will be removed", width="stretch")
            masks[name] = build_rect_mask(img.size, rect)

# ----------------------------------------------------------------------------
# Run inpainting
# ----------------------------------------------------------------------------

st.subheader("2. Remove")

if st.button("Remove watermarks", type="primary"):
    with st.spinner("Loading model (first run only)..."):
        model = load_model()

    results = {}
    progress = st.progress(0.0, text="Starting...")
    items = list(images.items())
    for i, (name, img) in enumerate(items):
        progress.progress(i / len(items), text=f"Processing {name}...")
        results[name] = run_inpaint(model, img, masks[name])
    progress.progress(1.0, text="Done.")

    st.session_state["results"] = results

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------

if "results" in st.session_state and st.session_state["results"]:
    st.subheader("3. Results")
    results = st.session_state["results"]

    zip_bytes = images_to_zip(results)
    st.download_button(
        "Download all as .zip",
        data=zip_bytes,
        file_name="watermark-removed.zip",
        mime="application/zip",
        type="primary",
    )

    cols = st.columns(2)
    for i, (name, result_img) in enumerate(results.items()):
        with cols[i % 2]:
            st.caption(name)
            show_original = st.toggle("Show original", value=False, key=f"toggle_{name}")
            st.image(images[name] if show_original else result_img, width="stretch")
