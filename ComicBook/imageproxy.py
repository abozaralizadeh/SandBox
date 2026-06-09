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

# Widths offered in the srcset. 480/768 cover phones; 1280 keeps desktop + retina
# crisp without shipping the full 1536px original.
ALLOWED_WIDTHS = (480, 768, 1280)
_WEBP_QUALITY = 80
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


def rewrite_comic_images(html: str) -> str:
    """Point panel <img> tags at the /cbimg WebP proxy with a responsive srcset.

    Only images inside our blob container are rewritten; anything else is left as
    is. The original PNG URL is preserved in `data-full` so the click-to-zoom modal
    can still show full resolution on demand.
    """
    if not html:
        return html
    prefix = _container_prefix()

    def repl(match: re.Match) -> str:
        url, rest = match.group(1), match.group(2)
        if not url.startswith(prefix):
            return match.group(0)
        enc = quote(url, safe="")
        srcset = ", ".join(f"/cbimg?u={enc}&w={w} {w}w" for w in ALLOWED_WIDTHS)
        sizes = "(max-width: 600px) 96vw, 480px"
        return (
            f'<img src="/cbimg?u={enc}&w=1280" srcset="{srcset}" sizes="{sizes}" '
            f'data-full="{url}" decoding="async"{rest}>'
        )

    return _IMG_RE.sub(repl, html)
