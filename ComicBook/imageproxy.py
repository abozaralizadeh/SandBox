"""On-the-fly WebP image proxy for ComicBook panels.

The originals are multi-MB PNGs (1024px wide for square/tall panels, 1536px for wide
ones) in the public blob container; they stay the canonical originals (they're reused
as references when generating later panels). For display we serve a single FULL-
RESOLUTION WebP per panel at high quality: WebP stays ~6-9× smaller than the PNG even
near-lossless, so downscaling for the web isn't worth the quality loss. The browser
scales the full-res image down to each panel's display size, which is sharp (only
*up*-scaling blurs).

  * `rewrite_comic_images(html)` rewrites each panel <img> to a `/cbimg?u=...` URL
    (called at serve time — the stored HTML keeps its PNG URLs, so the change is
    retroactive to every already-stored episode and the panel-reference chain that
    re-downloads prior panels is untouched).
  * `ensure_webp_variant(url, width, quality)` lazily transcodes + caches one derivative
    back into the same blob container and returns its public URL (used by /cbimg, which
    302-redirects the browser to the cached WebP).
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

# High quality by default — WebP stays far lighter than the PNG even near-lossless, so we
# favour fidelity over shaving kilobytes. Full native resolution is served (width=None).
DEFAULT_QUALITY = 92
_MAX_WIDTH = 4096  # safety clamp for the optional ?w override
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


def _derived_name(blob_name: str, width: int | None, quality: int) -> str:
    stem = blob_name.rsplit(".", 1)[0]
    wtag = f"_w{width}" if width else ""
    return f"{stem}{wtag}_q{quality}.webp"


def ensure_webp_variant(original_url: str, width: int | None = None, quality: int = DEFAULT_QUALITY) -> str | None:
    """Ensure a WebP of `original_url` exists in blob storage and return its public URL.

    Full native resolution unless `width` is given and smaller (we only ever downscale).
    Transcodes once; subsequent calls just confirm the cached blob exists. None if the
    URL isn't one of ours.
    """
    blob_name = blob_name_if_ours(original_url)
    if not blob_name:
        return None
    quality = max(1, min(100, quality))
    if width is not None:
        width = max(1, min(_MAX_WIDTH, width))

    derived = _derived_name(blob_name, width, quality)
    if blob_exists(derived):
        return _container_prefix() + derived

    try:
        img = Image.open(BytesIO(download_blob_bytes(blob_name))).convert("RGB")
        if width and img.width > width:  # only ever downscale
            height = round(img.height * width / img.width)
            img = img.resize((width, height), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="WEBP", quality=quality, method=6)
        return upload_blob_bytes(derived, buf.getvalue(), "image/webp")
    except Exception as exc:
        print(f"[ComicBook:imageproxy] transcode failed ({blob_name} w{width} q{quality}): {exc}")
        return None


def rewrite_comic_images(html: str) -> str:
    """Point panel <img> tags at the /cbimg full-resolution WebP proxy.

    Only images inside our blob container are rewritten; anything else is left as is. The
    original PNG URL is kept in `data-full` so the click-to-zoom modal shows the absolute
    original (lossless) on demand.
    """
    if not html:
        return html
    prefix = _container_prefix()

    def repl(match: re.Match) -> str:
        url, rest = match.group(1), match.group(2)
        if not url.startswith(prefix):
            return match.group(0)
        enc = quote(url, safe="")
        return f'<img src="/cbimg?u={enc}" data-full="{url}" decoding="async"{rest}>'

    return _IMG_RE.sub(repl, html)
