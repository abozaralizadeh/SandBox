"""On-the-fly WebP image proxy for ComicBook panels.

The original panels are multi-MB PNGs (1024-1536px) stored in the public blob
container; they stay the canonical originals (they're reused as references when
generating later panels). For *display* we serve resized WebP derivatives:

  * `rewrite_comic_images(html)` rewrites each panel <img> to a responsive srcset
    of `/cbimg?...` URLs (called at serve time — the stored HTML is untouched, so
    the change is retroactive to every already-stored episode).
  * `ensure_webp_variant(url, w)` lazily transcodes + caches one derivative back
    into the same blob container and returns its public URL (used by the /cbimg
    route, which 302-redirects the browser straight to the cached WebP).
"""

import re
from io import BytesIO
from urllib.parse import quote

from PIL import Image

from ComicBook.azurestorage import (
    get_blob_container_url,
    blob_exists,
    download_blob_bytes,
    upload_blob_bytes,
)

# Widths offered in the srcset, up to the originals' native widths (square/tall are
# 1024 wide, wide panels 1536) so the browser never has to upscale a too-small WebP.
# `ensure_webp_variant` never upscales, so e.g. w=1536 of a 1024px square just yields
# the 1024px source — the sharpest available for that panel.
ALLOWED_WIDTHS = (512, 768, 1024, 1536)
_WEBP_QUALITY = 82

# `sizes` = the panel's real CSS display width, so the browser (× devicePixelRatio)
# picks a high-enough srcset entry. Desktop: .comic-page is ~912px wide inner (max-width
# 960 − 2×24 padding); a 2-col grid makes square/tall panels ~451px and wide panels span
# the full ~912px. Mobile: every panel is full-width (single column).
_PANEL_SIZES = {
    "wide":   "(max-width: 600px) 96vw, 912px",
    "square": "(max-width: 600px) 96vw, 451px",
    "tall":   "(max-width: 600px) 96vw, 451px",
}
_DEFAULT_SIZES = _PANEL_SIZES["square"]

# Match a panel's wrapper + its <img> together, so we know the panel's size class.
_PANEL_IMG_RE = re.compile(
    r'(<div class="panel panel-(square|wide|tall)"[^>]*>)\s*<img\s+src="([^"]+?\.png)"([^>]*)>',
    re.IGNORECASE,
)
# Fallback for any panel image not caught above (unexpected wrapper).
_IMG_RE = re.compile(r'<img\s+src="([^"]+?\.png)"([^>]*)>', re.IGNORECASE)


def _container_prefix() -> str:
    return get_blob_container_url() + "/"


def blob_name_if_ours(url: str) -> str | None:
    """Return the blob name iff `url` points inside our container, else None.

    Guards the proxy against being used as an open relay/SSRF vector.
    """
    prefix = _container_prefix()
    if not url or not url.startswith(prefix):
        return None
    return url[len(prefix):].split("?", 1)[0]  # drop any query/SAS


def _derived_name(blob_name: str, width: int) -> str:
    stem = blob_name.rsplit(".", 1)[0]
    return f"{stem}_w{width}.webp"


def ensure_webp_variant(original_url: str, width: int) -> str | None:
    """Ensure a width-resized WebP of `original_url` exists in blob storage.

    Returns the derivative's public URL, or None if `original_url` isn't one of
    ours. Transcodes once (subsequent calls just confirm the cached blob exists).
    """
    blob_name = blob_name_if_ours(original_url)
    if not blob_name:
        return None
    if width not in ALLOWED_WIDTHS:
        width = min(ALLOWED_WIDTHS, key=lambda w: abs(w - width))

    derived = _derived_name(blob_name, width)
    if blob_exists(derived):
        return _container_prefix() + derived

    try:
        img = Image.open(BytesIO(download_blob_bytes(blob_name))).convert("RGB")
        if img.width > width:  # only ever downscale
            height = round(img.height * width / img.width)
            img = img.resize((width, height), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="WEBP", quality=_WEBP_QUALITY, method=6)
        return upload_blob_bytes(derived, buf.getvalue(), "image/webp")
    except Exception as exc:
        print(f"[ComicBook:imageproxy] transcode failed ({blob_name} w{width}): {exc}")
        return None


def _img_tag(url: str, rest: str, sizes: str) -> str:
    enc = quote(url, safe="")
    srcset = ", ".join(f"/cbimg?u={enc}&w={w} {w}w" for w in ALLOWED_WIDTHS)
    return (
        f'<img src="/cbimg?u={enc}&w=1024" srcset="{srcset}" sizes="{sizes}" '
        f'data-full="{url}" decoding="async"{rest}>'
    )


def rewrite_comic_images(html: str) -> str:
    """Point panel <img> tags at the /cbimg WebP proxy with a responsive srcset.

    `sizes` is set from each panel's size class so the browser picks a high-enough
    resolution (wide panels are ~2× the width of square/tall ones on desktop — getting
    this wrong is what made them upscale-blurry). Only images inside our blob container
    are rewritten; the original PNG URL is kept in `data-full` for full-res zoom.
    """
    if not html:
        return html
    prefix = _container_prefix()

    def panel_repl(match: re.Match) -> str:
        div, size, url, rest = match.group(1), match.group(2).lower(), match.group(3), match.group(4)
        if not url.startswith(prefix):
            return match.group(0)
        return div + _img_tag(url, rest, _PANEL_SIZES.get(size, _DEFAULT_SIZES))

    def img_repl(match: re.Match) -> str:
        url, rest = match.group(1), match.group(2)
        if not url.startswith(prefix):
            return match.group(0)
        return _img_tag(url, rest, _DEFAULT_SIZES)

    # Panel-aware pass first (gives wide panels their wider `sizes`); then a generic pass
    # for any leftover raw PNG <img> (already-rewritten tags no longer end in .png, so they
    # won't re-match).
    html = _PANEL_IMG_RE.sub(panel_repl, html)
    html = _IMG_RE.sub(img_repl, html)
    return html
