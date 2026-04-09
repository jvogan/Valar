#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "assets" / "media"

BG_TOP = "#0A1020"
BG_BOTTOM = "#111B33"
CARD = "#F8FAFC"
CARD_MUTED = "#E5E7EB"
INK = "#0F172A"
INK_SOFT = "#475569"
ACCENT = "#4F46E5"
ACCENT_2 = "#0F766E"
ACCENT_3 = "#C2410C"
PANEL = "#1E293B"
PANEL_SOFT = "#334155"
PANEL_TEXT = "#E2E8F0"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Neue Bold.ttf",
                "/System/Library/Fonts/SFNS.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Neue.ttc",
                "/System/Library/Fonts/SFNS.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def gradient(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, BG_TOP)
    draw = ImageDraw.Draw(image)
    top = tuple(int(BG_TOP[i : i + 2], 16) for i in (1, 3, 5))
    bottom = tuple(int(BG_BOTTOM[i : i + 2], 16) for i in (1, 3, 5))
    for y in range(height):
        t = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width, y), fill=color)
    return image


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str | None = None, radius: int = 28, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def pill(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, text_fill: str, font: ImageFont.ImageFont) -> int:
    left, top = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0] + 36
    height = bbox[3] - bbox[1] + 18
    draw.rounded_rectangle((left, top, left + width, top + height), radius=height // 2, fill=fill)
    draw.text((left + 18, top + 9), text, fill=text_fill, font=font)
    return width


def multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str, spacing: int = 12) -> None:
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), trial, font=font)
        if current and bbox[2] - bbox[0] > max_width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def draw_terminal_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, lines: list[tuple[str, str]]) -> None:
    left, top, right, bottom = box
    rounded(draw, box, fill=PANEL, outline="#23314F", radius=26, width=2)
    draw.rounded_rectangle((left, top, right, top + 48), radius=26, fill=PANEL_SOFT)
    for idx, color in enumerate(("#FB7185", "#FBBF24", "#34D399")):
        cx = left + 24 + idx * 20
        cy = top + 24
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color)
    title_font = load_font(22, bold=True)
    body_font = load_font(22)
    mono_font = load_font(20)
    draw.text((left + 110, top + 13), title, fill=PANEL_TEXT, font=title_font)
    y = top + 72
    for prefix, content in lines:
        draw.text((left + 24, y), prefix, fill="#93C5FD", font=mono_font)
        draw.text((left + 78, y), content, fill=PANEL_TEXT, font=body_font)
        y += 36


def draw_app_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    rounded(draw, box, fill=CARD, outline=CARD_MUTED, radius=28)
    header_h = 58
    draw.rounded_rectangle((left, top, right, top + header_h), radius=28, fill="#E2E8F0")
    for idx, color in enumerate(("#FB7185", "#FBBF24", "#34D399")):
        cx = left + 26 + idx * 20
        cy = top + 29
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color)
    title_font = load_font(24, bold=True)
    draw.text((left + 112, top + 16), "Valar macOS app", fill=INK, font=title_font)
    sidebar = (left + 18, top + 78, left + 186, bottom - 18)
    rounded(draw, sidebar, fill="#E5EEF9", radius=20)
    section_font = load_font(18, bold=True)
    item_font = load_font(17)
    draw.text((sidebar[0] + 18, sidebar[1] + 18), "Workspace", fill=INK_SOFT, font=section_font)
    items = ["Generator", "Models", "Voices", "Projects", "Diagnostics"]
    for idx, item in enumerate(items):
        y = sidebar[1] + 56 + idx * 44
        fill = "#D5E5FF" if idx == 0 else "#E5EEF9"
        rounded(draw, (sidebar[0] + 12, y - 8, sidebar[2] - 12, y + 26), fill=fill, radius=14)
        draw.text((sidebar[0] + 24, y), item, fill=INK, font=item_font)
    panel = (left + 206, top + 78, right - 18, bottom - 18)
    rounded(draw, panel, fill="#FFFFFF", radius=20)
    draw.text((panel[0] + 20, panel[1] + 18), "Generator", fill=INK, font=section_font)
    rounded(draw, (panel[0] + 20, panel[1] + 56, panel[2] - 20, panel[1] + 140), fill="#F8FAFC", outline="#E2E8F0", radius=18)
    message_font = load_font(20)
    message = wrap_text(
        draw,
        "Write text, audition voices, and export local audio without leaving your Mac.",
        message_font,
        panel[2] - panel[0] - 92,
    )
    multiline(draw, (panel[0] + 36, panel[1] + 76), message, message_font, INK)
    waveform_box = (panel[0] + 20, panel[1] + 168, panel[2] - 20, panel[1] + 292)
    rounded(draw, waveform_box, fill="#EEF2FF", radius=18)
    mid_y = (waveform_box[1] + waveform_box[3]) // 2
    x = waveform_box[0] + 24
    heights = [18, 34, 22, 50, 26, 40, 28, 54, 30, 38, 24, 44, 20, 32]
    for height in heights:
        draw.rounded_rectangle((x, mid_y - height, x + 16, mid_y + height), radius=8, fill=ACCENT)
        x += 24
    for idx, label in enumerate(("Soprano", "Qwen", "VibeVoice")):
        px = panel[0] + 20 + idx * 132
        pill(draw, (px, panel[1] + 314), label, "#E0E7FF", ACCENT, load_font(16, bold=True))
    rounded(draw, (panel[2] - 170, panel[1] + 312, panel[2] - 20, panel[1] + 352), fill=ACCENT_2, radius=20)
    draw.text((panel[2] - 140, panel[1] + 322), "Export WAV", fill="white", font=load_font(18, bold=True))


def draw_compact_app_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    rounded(draw, box, fill=CARD, outline=CARD_MUTED, radius=28)
    draw.rounded_rectangle((left, top, right, top + 58), radius=28, fill="#E2E8F0")
    for idx, color in enumerate(("#FB7185", "#FBBF24", "#34D399")):
        cx = left + 26 + idx * 20
        cy = top + 29
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color)
    draw.text((left + 112, top + 16), "Valar macOS app", fill=INK, font=load_font(22, bold=True))
    sidebar = (left + 18, top + 78, left + 186, bottom - 18)
    rounded(draw, sidebar, fill="#E5EEF9", radius=20)
    item_font = load_font(16)
    for idx, item in enumerate(("Generator", "Models", "Voices", "Projects")):
        y = sidebar[1] + 22 + idx * 44
        fill = "#D5E5FF" if idx == 0 else "#E5EEF9"
        rounded(draw, (sidebar[0] + 12, y - 6, sidebar[2] - 12, y + 22), fill=fill, radius=14)
        draw.text((sidebar[0] + 22, y), item, fill=INK, font=item_font)
    panel = (left + 206, top + 78, right - 18, bottom - 18)
    rounded(draw, panel, fill="#FFFFFF", radius=20)
    draw.text((panel[0] + 20, panel[1] + 18), "Generator", fill=INK, font=load_font(20, bold=True))
    rounded(draw, (panel[0] + 20, panel[1] + 54, panel[2] - 20, panel[1] + 134), fill="#F8FAFC", outline="#E2E8F0", radius=16)
    text = wrap_text(draw, "Write text, audition voices, and export local audio.", load_font(16), panel[2] - panel[0] - 86)
    multiline(draw, (panel[0] + 36, panel[1] + 72), text, load_font(16), INK, spacing=8)
    waveform_box = (panel[0] + 20, panel[1] + 156, panel[2] - 20, panel[1] + 244)
    rounded(draw, waveform_box, fill="#EEF2FF", radius=18)
    mid_y = (waveform_box[1] + waveform_box[3]) // 2
    x = waveform_box[0] + 20
    heights = [16, 30, 22, 38, 24, 34, 20, 40, 22, 32]
    for height in heights:
        draw.rounded_rectangle((x, mid_y - height, x + 14, mid_y + height), radius=7, fill=ACCENT)
        x += 22


def create_social_preview() -> Image.Image:
    image = gradient((1280, 640))
    draw = ImageDraw.Draw(image)
    big = load_font(74, bold=True)
    sub = load_font(28)
    chip = load_font(20, bold=True)
    body = load_font(30)
    draw.text((78, 76), "Valar", fill="white", font=big)
    multiline(
        draw,
        (78, 166),
        "Local speech stack for Apple Silicon\nTTS, ASR, forced alignment, voices,\ndaemon, and MCP bridge.",
        body,
        "#E2E8F0",
    )
    x = 80
    y = 350
    for text, fill in (
        ("TTS", ACCENT),
        ("ASR", ACCENT_2),
        ("Alignment", ACCENT_3),
        ("Voices", "#0EA5E9"),
    ):
        x += pill(draw, (x, y), text, fill, "white", chip) + 16
    draw_terminal_card(
        draw,
        (700, 88, 1210, 320),
        "CLI + daemon",
        [
            ("$", "make quickstart"),
            ("$", "make first-clip"),
            (">", "GET /v1/health"),
            ("{", '"status": "ok" }'),
        ],
    )
    draw_compact_app_card(draw, (700, 350, 1210, 594))
    return image


def create_hero() -> Image.Image:
    image = gradient((1600, 900))
    draw = ImageDraw.Draw(image)
    title = load_font(92, bold=True)
    sub = load_font(34)
    chip = load_font(22, bold=True)
    body = load_font(27)
    draw.text((96, 92), "Valar", fill="white", font=title)
    multiline(
        draw,
        (96, 208),
        "Local speech stack for Apple Silicon.\nGenerate speech, transcribe audio, align words,\nand expose it all through a daemon and MCP bridge.",
        body,
        "#E2E8F0",
    )
    x = 98
    y = 410
    for text, fill in (
        ("CLI first", ACCENT),
        ("Daemon", ACCENT_2),
        ("MCP bridge", "#0EA5E9"),
        ("App source", "#9333EA"),
    ):
        x += pill(draw, (x, y), text, fill, "white", chip) + 18
    multiline(
        draw,
        (96, 520),
        "Start with two commands, then grow into local HTTP,\nagent tooling, or the macOS app once the core stack works.",
        sub,
        "#CBD5E1",
        spacing=18,
    )
    draw_terminal_card(
        draw,
        (860, 84, 1490, 376),
        "CLI + MCP workflow",
        [
            ("$", "make quickstart"),
            ("$", "make first-clip"),
            ("$", "swift run --package-path apps/ValarDaemon valarttsd"),
            ("$", "cd bridge && bun server.ts"),
            ("{", '"tool": "valar_speak" }'),
        ],
    )
    draw_app_card(draw, (860, 422, 1490, 828))
    return image


def create_cli_mcp_preview() -> Image.Image:
    image = gradient((1600, 900))
    draw = ImageDraw.Draw(image)
    draw.text((86, 72), "CLI + MCP workflow", fill="white", font=load_font(60, bold=True))
    multiline(
        draw,
        (88, 154),
        "Build locally, start the daemon, then expose\nspeech tools to your agent through the MCP bridge.",
        load_font(30),
        "#D9E2F1",
    )
    draw_terminal_card(
        draw,
        (84, 280, 780, 760),
        "Terminal",
        [
            ("$", "make quickstart"),
            ("$", "make first-clip"),
            ("$", "swift run --package-path apps/ValarDaemon valarttsd"),
            ("$", "make bootstrap-bridge"),
            ("$", "cd bridge && bun server.ts"),
        ],
    )
    rounded(draw, (846, 280, 1510, 760), fill=CARD, outline=CARD_MUTED, radius=28)
    draw.text((876, 316), "MCP tool view", fill=INK, font=load_font(34, bold=True))
    pill(draw, (876, 374), "Tool call", "#E0E7FF", ACCENT, load_font(18, bold=True))
    pill(draw, (1022, 374), "Local only", "#DCFCE7", ACCENT_2, load_font(18, bold=True))
    rounded(draw, (876, 430, 1476, 522), fill="#F8FAFC", outline="#E2E8F0", radius=18)
    draw.text((904, 452), "valar_speak", fill=INK, font=load_font(26, bold=True))
    draw.text((904, 488), 'model: "mlx-community/Soprano-1.1-80M-bf16"', fill=INK_SOFT, font=load_font(22))
    rounded(draw, (876, 552, 1476, 720), fill="#EEF2FF", radius=22)
    draw.text((904, 578), "Result", fill=ACCENT, font=load_font(24, bold=True))
    multiline(
        draw,
        (904, 620),
        "WAV generated locally.\nReady for playback, automation,\nor the next agent step.",
        load_font(28),
        INK,
    )
    return image


def create_app_preview() -> Image.Image:
    image = gradient((1600, 900))
    draw = ImageDraw.Draw(image)
    draw.text((86, 72), "macOS app preview", fill="white", font=load_font(60, bold=True))
    multiline(
        draw,
        (88, 154),
        "The desktop UI stays secondary to the CLI path,\nbut the same local runtime powers the app surface too.",
        load_font(30),
        "#D9E2F1",
    )
    draw_app_card(draw, (84, 258, 1516, 804))
    return image


def save_png(name: str, image: Image.Image) -> None:
    path = MEDIA_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def main() -> None:
    save_png("social-preview.png", create_social_preview())
    save_png("hero.png", create_hero())
    save_png("cli-mcp-preview.png", create_cli_mcp_preview())
    save_png("app-preview.png", create_app_preview())


if __name__ == "__main__":
    main()
