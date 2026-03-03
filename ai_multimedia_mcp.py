import os
import shutil
import httpx
import subprocess
import json
import re

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None
from pathlib import Path
from typing import Optional, Literal
from mcp.server.fastmcp import FastMCP, Context
from google import genai
from google.genai import types
import fal_client

# Initialize FastMCP server
mcp = FastMCP("ai-multimedia-supreme")

# Fixed backup directory for all generated media
MEDIA_ARCHIVE_DIR = Path.home() / "Documents" / "ai-multimedia-files"
FAL_DOCS_MCP_URL = "https://docs.fal.ai/mcp"


def get_google_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def check_fal_key():
    if not os.getenv("FAL_KEY"):
        raise ValueError("FAL_KEY environment variable is not set.")


async def download_file(url: str, output_path: Path) -> None:
    """Downloads a file from a URL and saves it locally."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(follow_redirects=True) as http:
        response = await http.get(url, timeout=120.0)
        response.raise_for_status()
        output_path.write_bytes(response.content)


async def ensure_public_url(path_or_url: str) -> str:
    """If input is a local file path, upload to Fal.ai and return public URL. URLs pass through."""
    if path_or_url.startswith(("https://", "http://", "data:")):
        return path_or_url
    check_fal_key()
    local_path = Path(path_or_url).resolve()
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    return await fal_client.upload_file_async(str(local_path))


def resolve_output_path(output_filename: str) -> Path:
    """
    Resolves the primary output path.
    - Absolute paths are used as-is.
    - Relative paths resolve to the current working directory.
    """
    p = Path(output_filename)
    path = p if p.is_absolute() else Path.cwd() / p.name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def copy_to_archive(output_path: Path) -> Path:
    """
    Copies the generated file to ~/Documents/ai-multimedia-files/ as a permanent backup.
    Adds a numeric suffix if a file with the same name already exists.
    """
    MEDIA_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = MEDIA_ARCHIVE_DIR / output_path.name
    if archive_path.exists() and archive_path != output_path:
        stem = archive_path.stem
        suffix = archive_path.suffix
        counter = 1
        while archive_path.exists():
            archive_path = MEDIA_ARCHIVE_DIR / f"{stem}_{counter}{suffix}"
            counter += 1
    if output_path != archive_path:
        shutil.copy2(output_path, archive_path)
    return archive_path


def format_result(media_type: str, engine: str, local_path: str, public_url: str, archive_path: str = "") -> str:
    """Standard return format with local path, public URL, and archive copy."""
    lines = [
        f"Successfully generated {media_type} via {engine}.",
        f"LOCAL_PATH: {local_path}",
        f"PUBLIC_URL: {public_url}",
    ]
    if archive_path:
        lines.append(f"ARCHIVE_COPY: {archive_path}")
    lines.append("Use PUBLIC_URL when passing this asset to other tools (e.g. create_talking_avatar, generate_video).")
    return "\n".join(lines)


# =============================================================================
# FAL DOCS BRIDGE (embedded docs consultation)
# =============================================================================

@mcp.tool()
async def consult_fal_docs(
    topic: str,
    model_hint: Optional[str] = None,
    max_chars: int = 12000,
    ctx: Context = None
) -> str:
    """
    Consults Fal documentation sources before generation/update decisions.

    Strategy:
    1) Try Fal Docs MCP endpoint (`https://docs.fal.ai/mcp`) via JSON-RPC best-effort
       to discover available docs tools.
    2) Fallback to fetching human docs pages from docs.fal.ai.

    Use this tool before complex model routing or when verifying latest endpoints/pricing.
    """
    snippets = []
    if ctx:
        await ctx.info("Consulting Fal docs sources...")

    # 1) Best-effort MCP introspection
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ai-multimedia-supreme", "version": "1.0"}
                }
            }
            init_res = await http.post(FAL_DOCS_MCP_URL, json=init_payload)
            init_text = init_res.text[:1200]
            snippets.append(f"[Fal Docs MCP initialize]\nstatus={init_res.status_code}\n{init_text}")

            list_payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            list_res = await http.post(FAL_DOCS_MCP_URL, json=list_payload)
            list_text = list_res.text[:3000]
            snippets.append(f"[Fal Docs MCP tools/list]\nstatus={list_res.status_code}\n{list_text}")
    except Exception as e:
        snippets.append(f"[Fal Docs MCP unavailable] {e}")

    # 2) Fallback web docs pages
    docs_urls = [
        "https://docs.fal.ai/",
        "https://docs.fal.ai/model-apis",
        "https://docs.fal.ai/model-apis/guides",
    ]
    if model_hint:
        docs_urls.append(f"https://docs.fal.ai/model-apis/{model_hint}")

    for url in docs_urls:
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http:
                r = await http.get(url)
                raw_html = r.text
                if BeautifulSoup is not None:
                    text = BeautifulSoup(raw_html, "html.parser").get_text(separator="\n", strip=True)
                else:
                    # Fallback text extraction without bs4
                    text = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
                    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()

                # topical extraction on cleaned text
                if topic.lower() in text.lower() or len(snippets) < 5:
                    snippets.append(f"[Docs: {url}]\nstatus={r.status_code}\n{text[:2200]}")
        except Exception as e:
            snippets.append(f"[Docs fetch failed: {url}] {e}")

    joined = "\n\n".join(snippets)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n...[truncated]"

    return (
        "FAL_DOCS_CONSULTED\n"
        f"topic={topic}\n"
        f"model_hint={model_hint or 'none'}\n\n"
        f"{joined}\n\n"
        "Use this information to verify model IDs/endpoints before generation."
    )


# =============================================================================
# CONSULTATION TOOL — Must be called first to discuss options with the user
# =============================================================================

@mcp.tool()
async def consult_multimedia_options(
    media_type: Literal["image", "video", "audio", "avatar"]
) -> str:
    """
    MANDATORY FIRST STEP — Call this BEFORE any generation tool.
    Returns all available options, parameters, and pricing for the requested media type.

    After receiving the options, you MUST have a conversation with the user to determine:
    - Which engine/model they want (quality vs price tradeoff)
    - All creative parameters (aspect ratio, voice, style, duration, etc.)
    - Budget confirmation for paid models

    DO NOT call generate tools until the user has explicitly confirmed their choices.
    DO NOT assume defaults — ask the user for EVERY creative decision.
    """
    global_rule = """
=== GLOBAL CRITICAL RULE (ALL MEDIA TYPES) ===
The user does NOT have to generate every asset with AI.
Always ask first: "Do you want to generate assets from scratch, or use your own existing local files?"
You can accept absolute local file paths (e.g., /home/user/video.mp4 or /home/user/voice.m4a)
and pass them directly into tools like merge_audio_video or create_talking_avatar.

FAL DOCS ENFORCEMENT:
Before finalizing model choice or endpoint routing (especially after recent updates), call consult_fal_docs.
Use docs findings to adapt the plan and avoid outdated/removed endpoints.
"""

    if media_type == "image":
        return f"""
{global_rule}
=== IMAGE GENERATION OPTIONS ===

ENGINES (ask the user which one):
1. "nano-banana" — Google Gemini 2.5 Flash Image
   Price: Free/very cheap | Speed: Very fast | Best for: Drafts, iterations, concepts
2. "nano-banana-pro" — Google Gemini 3 Pro Image Preview
   Price: Low | Speed: Fast | Best for: Professional assets, advanced reasoning
3. "flux-schnell" — FLUX Schnell (Fal.ai)
   Price: $0.015/image | Speed: Fast | Best for: UI elements, icons, general use
4. "flux-pro" — FLUX Pro 1.1 (Fal.ai)
   Price: $0.05+/image | Speed: Medium | Best for: Photorealism, text in images, logos
5. "flux-2-max" — FLUX 2 Max (Fal.ai)
   Price: $0.08+/image | Speed: Slower | Best for: Absurd photorealism, perfect text, complex prompts

QUESTIONS YOU MUST ASK THE USER:
- What is the image for? (web banner, avatar, logo, social media, app icon, etc.)
- Which engine? Explain the price/quality tradeoff.
- What aspect ratio? Options: 1:1 (square/icon), 16:9 (landscape/web), 9:16 (portrait/mobile), 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 21:9 (ultrawide)
- What style? (photorealistic, illustration, cartoon, minimalist, cinematic, etc.)
- For flux-pro/max: Enhanced prompt optimization? (enhance_prompt)
- Need reproducibility? (seed for exact same result)
- Output format: PNG (lossless) or JPEG (compressed)?

OUTPUT: Saves to current working directory AND backup copy to ~/Documents/ai-multimedia-files/
"""
    elif media_type == "video":
        return f"""
{global_rule}
=== VIDEO GENERATION OPTIONS ===

NEW: Video-to-Video (Restyling/Transformation) is now supported.
You can transform an existing video into a new style (e.g., "Transform the person into a 3D claymation character").
If the user provides a local MP4 path, the system can auto-upload and process it.

ENGINES (ask the user which one):
1. "kling" — Kling 3.0 Pro (Fal.ai)
   Price: ~$0.10/sec (varies by mode) | Best for: Cinematic quality, character consistency, realistic physics
   Supports: Native audio, negative prompts, CFG control. Duration: 5s, 10s, 15s.
2. "runway" — Runway Gen-4 Turbo (Fal.ai)
   Price: ~$0.05/sec | Best for: Creative control, high fidelity, speed
3. "minimax" — MiniMax Hailuo 2.3 (Fal.ai)
   Price: $0.28/6sec or $0.56/10sec | Best for: Great motion physics, camera control
   Supports: Camera movements in prompt: [Pan left], [Zoom in], [Tracking shot], [Tilt up], etc.

COST STRATEGY DECISION TREE (for "video with sound"):
A) Talking Head Scenario (~$0.15)
   - Person talking to camera.
   - DO NOT use regular video generation.
   - Use Avatar workflow: Image -> TTS -> create_talking_avatar.

B) Cinematic/B-Roll Scenario (~$0.30)
   - General scene + voiceover/music.
   - Generate mute video with generate_video(generate_audio=False),
     generate audio separately with generate_audio, then merge with merge_audio_video.

C) Complex Physics/Lip-sync (~$1.30+)
   - Exact physical sync or complex interaction requiring native model audio alignment.
   - Use Kling with generate_audio=True.
   - MUST warn user this path is significantly more expensive before generation.

QUESTIONS YOU MUST ASK THE USER:
- Is this Text-to-Video, Image-to-Video, or Video-to-Video?
- If Video-to-Video: ask for an input video (PUBLIC_URL or absolute local MP4 path) and confirm the desired transformation style.
- Which engine? Explain the price/quality differences. Explicitly state Minimax is not available for Video-to-Video.
- Duration: 5, 10, or 15 seconds (Kling only)? (videos are expensive — confirm budget)
- Aspect ratio: 16:9 (landscape), 9:16 (vertical/mobile), 1:1 (square)?
- For Kling: Native audio generation? (doubles cost but adds voiceover/sounds)
- For Kling: Any negative prompt? (things to avoid)
- For Hailuo: Any camera movements? ([Pan left], [Zoom in], [Pull out], [Static shot], etc.)
- Confirm total estimated cost before generating.
- IMPORTANT: You must call generate_video once with confirm_cost=false first, show the estimate to the user, and only proceed after explicit user confirmation.
- If the plan involves merging audio and video, explicitly ask: "Do you want the video to loop if the audio is longer, or should I cut the final file to the shortest media?" Pass their choice to loop_video.
"""
    elif media_type == "audio":
        return f"""
{global_rule}
=== AUDIO GENERATION OPTIONS ===

TYPE 1: "tts" — Text-to-Speech (ElevenLabs Multilingual v2)
  Price: ~$0.01/request | 29 languages | Auto-accent detection
  VOICES — Ask the user which one:
    Female: Aria (warm, conversational), Sarah (clear, professional), Laura (expressive, dramatic),
            Charlotte (elegant, refined), Jessica (energetic, upbeat), Lily (soft, gentle),
            Bella (Soft, American/Latina friendly)
    Male:   Roger (confident, authoritative), Charlie (casual, friendly), George (deep, mature),
            Liam (young, dynamic), Daniel (calm, neutral), Chris (warm, natural),
            Adam (Deep, American/Latino friendly)
  PARAMETERS to discuss:
    - Speed: 0.7 (slow/dramatic) to 1.2 (fast/energetic). Default 1.0.
    - Stability: 0.0 (very expressive/variable) to 1.0 (very stable/consistent). Default 0.5.
    - Similarity boost: 0.0 to 1.0. Higher = more similar to original voice. Default 0.75.
    - Style exaggeration: 0.0 to 1.0. Higher = more expressive/dramatic.
    - Language: Auto-detected, or force specific (ISO 639-1: "es", "en", "fr", etc.)

TYPE 2: "sfx" — Sound Effects (ElevenLabs Sound Effects v2)
  Price: $0.002/second | Duration: 0.5-22 seconds
  PARAMETERS to discuss:
    - What specific sound? (explosion, whoosh, UI click, rain, footsteps, etc.)
    - Duration in seconds?
    - Prompt influence: 0.0 (creative) to 1.0 (strict). Default 0.3.

TYPE 3: "music" — Music Generation (ElevenLabs Music)
  Price: $0.80/minute | Duration: 3-600 seconds
  PARAMETERS to discuss:
    - Genre/mood? (cinematic, electronic, jazz, ambient, rock, lo-fi, epic, etc.)
    - Instrumental only or allow vocals?
    - Duration in seconds?
    - Detailed composition plan with sections (intro, verse, chorus, outro)?

QUESTIONS YOU MUST ASK THE USER:
- Voiceover (tts), sound effect (sfx), or music?
- For TTS: Which voice? Male or female? What tone/style?
- For TTS: Language? Speed? More expressive or more stable?
- For SFX: What specific sound? How long?
- For Music: What genre? Instrumental or vocals? How long?
"""
    elif media_type == "avatar":
        return f"""
{global_rule}
=== TALKING AVATAR OPTIONS ===

ENGINES (ask the user which one):
1. "live-avatar" — Live Avatar (Fal.ai): Natural real-time expressions.
2. "kling-avatar" — Kling AI Avatar v2 (Fal.ai): Studio-grade lipsync.

REQUIREMENTS:
- image_url: Portrait image (PUBLIC_URL or local path — auto-uploaded if local)
- audio_url: Speech audio (PUBLIC_URL or local path — auto-uploaded if local)

WORKFLOW — Explain to the user:
1. Generate or provide a portrait image → use generate_image → get PUBLIC_URL
2. Generate speech audio → use generate_audio(engine="tts") → get PUBLIC_URL
3. Call create_talking_avatar with both PUBLIC_URLs (or local paths)

QUESTIONS YOU MUST ASK THE USER:
- Do they have a portrait image, or should we generate one?
- What should the avatar say?
- Which voice for the speech?
- Which avatar engine? live-avatar (natural) or kling-avatar (studio quality)?
"""
    return "Invalid media type. Choose: image, video, audio, or avatar."


# =============================================================================
# FILE UPLOAD TOOL
# =============================================================================

@mcp.tool()
async def upload_file(
    file_path: str,
    ctx: Context = None
) -> str:
    """
    Uploads a local file to Fal.ai cloud and returns a public HTTPS URL.
    Use this when you need a public URL for a local file.
    """
    try:
        check_fal_key()
        local_path = Path(file_path).resolve()
        if not local_path.exists():
            return f"Error: File not found at {local_path}"
        if ctx:
            await ctx.info(f"Uploading {local_path.name}...")
        url = await fal_client.upload_file_async(str(local_path))
        return f"Successfully uploaded.\nLOCAL_PATH: {local_path}\nPUBLIC_URL: {url}"
    except Exception as e:
        return f"Failed to upload file: {str(e)}"


# =============================================================================
# LOCAL MERGE TOOL (FFMPEG)
# =============================================================================

@mcp.tool()
async def merge_audio_video(
    video_path: str,
    audio_path: str,
    output_filename: str,
    loop_video: bool = False,
    ctx: Context = None
) -> str:
    """
    Merges a local video file and a local audio file using local FFmpeg.

    Requirements:
    - video_path and audio_path must be local existing files.
    - If loop_video is True, video is looped infinitely, and final output is cut by -shortest.

    FFmpeg behavior:
    - Uses `-c:v copy` and `-c:a aac`
    - Uses `-shortest` to stop at the shortest active stream.
    """
    try:
        check_fal_key()

        # Resolve using existing path helper as requested
        resolved_video_path = resolve_output_path(video_path)
        resolved_audio_path = resolve_output_path(audio_path)
        output_path = resolve_output_path(output_filename)

        if not resolved_video_path.exists():
            return f"Error: video file not found at {resolved_video_path}"
        if not resolved_audio_path.exists():
            return f"Error: audio file not found at {resolved_audio_path}"

        ffmpeg_cmd = ["ffmpeg", "-y"]
        if loop_video:
            ffmpeg_cmd.extend(["-stream_loop", "-1"])

        ffmpeg_cmd.extend([
            "-i", str(resolved_video_path),
            "-i", str(resolved_audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_path),
        ])

        if ctx:
            await ctx.info("Merging local video+audio with FFmpeg...")

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)

        if ctx:
            await ctx.info("Uploading merged output to Fal storage...")

        public_url = await fal_client.upload_file_async(str(output_path))
        archive = copy_to_archive(output_path)
        return format_result("merged media", "ffmpeg", str(output_path), public_url, str(archive))

    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        return f"Failed to merge with ffmpeg: {stderr or str(e)}"
    except Exception as e:
        return f"Failed to merge audio+video: {str(e)}"


# =============================================================================
# IMAGE GENERATION
# =============================================================================

@mcp.tool()
async def generate_image(
    prompt: str,
    output_filename: str,
    engine: Literal["nano-banana", "nano-banana-pro", "flux-schnell", "flux-pro", "flux-2-max"],
    aspect_ratio: Literal["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
    seed: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_images: int = 1,
    output_format: Literal["png", "jpeg"] = "png",
    enhance_prompt: bool = False,
    safety_tolerance: Optional[int] = None,
    ctx: Context = None
) -> str:
    """
    Generates an image. Returns LOCAL_PATH, PUBLIC_URL, and ARCHIVE_COPY.

    REQUIRED — engine and aspect_ratio have NO defaults. Ask the user before calling.
    Call consult_multimedia_options("image") first to discuss options with the user.

    Engines:
    - nano-banana: Gemini 2.5 Flash Image (free, fast drafts)
    - nano-banana-pro: Gemini 3 Pro Image Preview (higher quality)
    - flux-schnell: FLUX Schnell ($0.015, fast, good general use)
    - flux-pro: FLUX Pro 1.1 ($0.05+, best photorealism, perfect text rendering)
    - flux-2-max: FLUX 2 Max ($0.08+, state-of-the-art photorealism)

    Advanced params (Flux only):
    - seed: Reproducible results. Same seed + same prompt = same image.
    - guidance_scale: 1-20, prompt adherence. Higher = more literal. Default ~3.5.
    - enhance_prompt: Auto-optimize prompt (flux-pro/max only).
    - safety_tolerance: 1 (strict) to 6 (permissive). Default 2.
    - num_images: 1-4 images per call.
    """
    try:
        output_path = resolve_output_path(output_filename)

        if "nano-banana" in engine:
            client = get_google_client()
            model = "gemini-2.5-flash-image" if engine == "nano-banana" else "gemini-3-pro-image-preview"

            config = types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
            )

            if ctx:
                await ctx.info(f"Generating image with Google '{model}'...")

            response = await client.aio.models.generate_content(
                model=model,
                contents=[prompt],
                config=config
            )

            if response.parts and response.parts[0].inline_data:
                img = response.parts[0].as_image()
                img.save(output_path)
                if ctx:
                    await ctx.info("Uploading to cloud for public URL...")
                check_fal_key()
                public_url = await fal_client.upload_file_async(str(output_path))
                archive = copy_to_archive(output_path)
                return format_result("image", engine, str(output_path), public_url, str(archive))
            return "Failed: Google returned no image data."

        elif "flux" in engine:
            check_fal_key()
            # Map engine names to Fal endpoint IDs
            if engine == "flux-schnell":
                model = "fal-ai/flux/schnell"
            elif engine == "flux-2-max":
                model = "fal-ai/flux-2-max/text-to-image" # New 2026 model
            else:
                model = "fal-ai/flux-pro/v1.1"

            size_map = {
                "16:9": "landscape_16_9", "9:16": "portrait_16_9",
                "4:3": "landscape_4_3", "3:4": "portrait_4_3",
                "1:1": "square_hd",
            }
            fal_image_size = size_map.get(aspect_ratio, "square_hd")

            arguments = {
                "prompt": prompt,
                "image_size": fal_image_size,
                "num_images": num_images,
                "output_format": output_format,
            }
            if seed is not None:
                arguments["seed"] = seed
            if guidance_scale is not None:
                arguments["guidance_scale"] = guidance_scale
            if engine in ["flux-pro", "flux-2-max"]:
                arguments["enhance_prompt"] = enhance_prompt
                if safety_tolerance is not None:
                    arguments["safety_tolerance"] = safety_tolerance

            if ctx:
                await ctx.info(f"Generating {num_images} image(s) with Fal.ai '{model}'...")

            result = await fal_client.subscribe_async(
                model, arguments=arguments, with_logs=True, client_timeout=120.0
            )

            if "images" in result and result["images"]:
                public_url = result["images"][0]["url"]
                await download_file(public_url, output_path)
                archive = copy_to_archive(output_path)
                extra = ""
                if len(result["images"]) > 1:
                    extra_urls = [img["url"] for img in result["images"][1:]]
                    extra = f"\nADDITIONAL_IMAGES: {', '.join(extra_urls)}"
                return format_result("image", engine, str(output_path), public_url, str(archive)) + extra
            return "Failed: Fal.ai returned no image data."

    except Exception as e:
        return f"Failed to generate image: {str(e)}"


# =============================================================================
# VIDEO GENERATION
# =============================================================================

@mcp.tool()
async def generate_video(
    prompt: str,
    output_filename: str,
    engine: Literal["kling", "runway", "minimax"],
    duration: Literal[5, 6, 10, 15],
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9",
    image_url: Optional[str] = None,
    input_video_url: Optional[str] = None,
    generate_audio: bool = False,
    negative_prompt: Optional[str] = None,
    cfg_scale: Optional[float] = None,
    prompt_optimizer: bool = True,
    confirm_cost: bool = False,
    acknowledged_estimated_cost_usd: Optional[float] = None,
    max_budget_usd: float = 2.0,
    ctx: Context = None
) -> str:
    """
    Generates a video from text or image. Returns LOCAL_PATH, PUBLIC_URL, and ARCHIVE_COPY.

    REQUIRED — engine and duration have NO defaults. Ask the user before calling.
    Call consult_multimedia_options("video") first to discuss options with the user.

    Engines:
    - kling: Kling 3.0 Pro (Fal.ai). Supports text-to-video, image-to-video, and video-to-video.
    - runway: Runway Gen-4 Turbo (Fal.ai). Supports text-to-video, image-to-video, and video-to-video.
    - minimax: Hailuo 2.3 ($0.28/6sec, $0.56/10sec). Supports text-to-video and image-to-video.

    Inputs:
    - image_url: for image-to-video (PUBLIC_URL from generate_image, or local path)
    - input_video_url: for video-to-video restyling/transformation (PUBLIC_URL or local path)

    Advanced params:
    - generate_audio: Native audio for Kling (English/Chinese). Doubles cost.
    - negative_prompt: What to avoid (Kling only).
    - cfg_scale: 0-1, prompt adherence for Kling. Default 0.5.
    - prompt_optimizer: Auto-optimize prompt for Hailuo. Default True.

    COST SAFETY (mandatory):
    - First call MUST use confirm_cost=False (default). The tool will return an estimate and stop.
    - Show the estimate to the user and ask for explicit confirmation.
    - Second call must set confirm_cost=True and pass acknowledged_estimated_cost_usd with the shown estimate.
    - max_budget_usd (default 2.0): if estimated cost exceeds this value, generation is blocked immediately.
      To proceed, explicitly raise max_budget_usd in your call.

    Hailuo camera movements (include in prompt): [Pan left], [Zoom in], [Tracking shot],
    [Tilt up/down], [Push in], [Pull out], [Pedestal up/down], [Truck left/right], [Static shot].
    """
    try:
        check_fal_key()
        output_path = resolve_output_path(output_filename)
        model = ""

        # Preflight cost estimate (required before generation)
        estimated_cost_usd = 0.0
        if engine == "kling":
            # Rough estimate based on current public pricing behavior
            estimated_cost_usd = (duration / 5.0) * (0.35 if generate_audio else 0.15)
        elif engine == "runway":
            estimated_cost_usd = (duration if duration in [5, 10] else 10) * 0.05
        elif engine == "minimax":
            estimated_cost_usd = 0.28 if duration <= 6 else 0.56

        if estimated_cost_usd > max_budget_usd:
            return (
                "BUDGET_EXCEEDED\n"
                f"estimated_cost_usd={estimated_cost_usd:.2f}\n"
                f"max_budget_usd={max_budget_usd:.2f}\n\n"
                "Generation blocked by budget guardrail. "
                "If the user explicitly accepts a higher budget, call again with a higher max_budget_usd."
            )

        if not confirm_cost:
            return (
                "COST_ESTIMATE_REQUIRED\n"
                f"engine={engine}\n"
                f"duration={duration}s\n"
                f"generate_audio={generate_audio}\n"
                f"estimated_cost_usd={estimated_cost_usd:.2f}\n"
                f"max_budget_usd={max_budget_usd:.2f}\n\n"
                "This is a paid generation. Show this estimate to the user and ask for explicit confirmation. "
                "Then call generate_video again with confirm_cost=true and acknowledged_estimated_cost_usd set to this exact estimate."
            )

        if acknowledged_estimated_cost_usd is None:
            return (
                "CONFIRMATION_MISSING\n"
                "You set confirm_cost=true but did not provide acknowledged_estimated_cost_usd. "
                "Pass the estimate returned by the preflight call."
            )

        if abs(acknowledged_estimated_cost_usd - estimated_cost_usd) > 0.011:
            return (
                "CONFIRMATION_MISMATCH\n"
                f"current_estimated_cost_usd={estimated_cost_usd:.2f}\n"
                f"acknowledged_estimated_cost_usd={acknowledged_estimated_cost_usd:.2f}\n"
                "Reconfirm with the user using the current estimate, then retry."
            )

        if image_url:
            image_url = await ensure_public_url(image_url)
        if input_video_url:
            input_video_url = await ensure_public_url(input_video_url)

        arguments = {"prompt": prompt}

        if engine == "kling":
            # Kling supports T2V / I2V / V2V
            if input_video_url:
                model = "fal-ai/kling-video/v3/pro/video-to-video"
                arguments["video_url"] = input_video_url
            else:
                model = "fal-ai/kling-video/v3/pro/image-to-video" if image_url else "fal-ai/kling-video/v3/pro/text-to-video"
                if image_url:
                    arguments["image_url"] = image_url
            arguments["duration"] = duration
            arguments["aspect_ratio"] = aspect_ratio
            if generate_audio:
                arguments["generate_audio"] = True
            if negative_prompt:
                arguments["negative_prompt"] = negative_prompt
            if cfg_scale is not None:
                arguments["cfg_scale"] = cfg_scale

        elif engine == "runway":
            # Runway supports T2V / I2V / V2V
            if input_video_url:
                model = "fal-ai/runway-gen3/video-to-video"
                arguments["video_url"] = input_video_url
            else:
                model = "fal-ai/runway-gen3/image-to-video" if image_url else "fal-ai/runway-gen3/text-to-video"
                if image_url:
                    arguments["image_url"] = image_url

        elif engine == "minimax":
            if input_video_url:
                return "Minimax currently does not support Video-to-Video. Please use Kling or Runway."
            model = "fal-ai/minimax/hailuo-2.3/standard/image-to-video" if image_url else "fal-ai/minimax/hailuo-2.3/standard/text-to-video"
            if image_url:
                arguments["image_url"] = image_url
            arguments["duration"] = str(6 if duration <= 6 else 10)
            arguments["prompt_optimizer"] = prompt_optimizer
        else:
            return "Invalid video engine."

        if ctx:
            cost_hint = f" (confirmed estimate: ~${estimated_cost_usd:.2f})"
            await ctx.info(f"Generating video with '{model}'{cost_hint}... This may take 30-900 seconds depending on queue load.")
            await ctx.report_progress(progress=0.1, total=1.0)

        result = await fal_client.subscribe_async(
            model, arguments=arguments, with_logs=True, client_timeout=900.0
        )

        if ctx:
            await ctx.report_progress(progress=0.9, total=1.0)

        if "video" in result:
            public_url = result["video"]["url"]
            await download_file(public_url, output_path)
            archive = copy_to_archive(output_path)
            return format_result("video", engine, str(output_path), public_url, str(archive))
        return f"Unexpected response from {engine}: {str(result)}"

    except Exception as e:
        err = str(e)
        if "timed out" in err and "Request " in err:
            return (
                f"Failed to generate video: {err}\n"
                "Tip: provider queue is congested. Re-run later or switch engine. "
                "If a request_id appears above, keep it for support/recredit."
            )
        return f"Failed to generate video: {err}"


# =============================================================================
# AUDIO GENERATION
# =============================================================================

@mcp.tool()
async def generate_audio(
    prompt: str,
    output_filename: str,
    engine: Literal["tts", "sfx", "music"],
    voice: Literal[
        "Aria", "Sarah", "Laura", "Charlotte", "Jessica", "Lily",
        "Roger", "Charlie", "George", "Liam", "Daniel", "Chris",
        "Bella", "Adam", "Jennifer", "Will"
    ] = "Aria",
    speed: float = 1.0,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    language_code: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    prompt_influence: float = 0.3,
    force_instrumental: bool = False,
    ctx: Context = None
) -> str:
    """
    Generates audio. Returns LOCAL_PATH, PUBLIC_URL, and ARCHIVE_COPY.

    REQUIRED — engine has NO default. Ask the user what type of audio they need.
    Call consult_multimedia_options("audio") first to discuss options with the user.

    engine="tts" — Text-to-Speech (ElevenLabs Multilingual v2):
      Ask the user which voice. Voices: Aria (warm F), Sarah (clear F), Laura (expressive F),
      Charlotte (elegant F), Jessica (energetic F), Lily (soft F), Roger (confident M),
      Charlie (casual M), George (deep M), Liam (young M), Daniel (calm M), Chris (warm M).
      Special LatAm: Bella (Soft F), Adam (Deep M).
      - speed: 0.7 (slow) to 1.2 (fast). Ask the user.
      - stability: 0.0 (expressive) to 1.0 (consistent). Ask for tone preference.
      - similarity_boost: 0.0 to 1.0. Closer to original voice character.
      - style: 0.0 to 1.0. More dramatic/expressive delivery.
      - language_code: ISO 639-1 ("es", "en", "fr", "pt", etc.) or auto-detect.

    engine="sfx" — Sound Effects (ElevenLabs v2, $0.002/sec):
      - duration_seconds: 0.5-22 sec. Ask what sound and how long.
      - prompt_influence: 0.0 (creative) to 1.0 (literal). Default 0.3.

    engine="music" — Music (ElevenLabs Music, $0.80/min):
      - duration_seconds: 3-600 sec. Ask genre, mood, and duration.
      - force_instrumental: True = no vocals. Ask the user.
    """
    try:
        check_fal_key()
        output_path = resolve_output_path(output_filename)
        model = ""
        arguments: dict = {}

        if engine == "tts":
            model = "fal-ai/elevenlabs/tts/multilingual-v2"
            
            # Map voice names to IDs (default to Aria if unknown)
            voice_map = {
                "Bella": "EXAVITQu4vr4xnSDxMaL", # American/Latina
                "Adam": "pNInz6obpgDQGcFmaJgB", # Deep American/Latino
                "Aria": "9BWtsMINqrJLrRacOk9x", 
                "Roger": "CwhRBWXzGAHq8TQ4Fs17",
                "Sarah": "EXAVITQu4vr4xnSDxMaL", # Using Bella ID for Sarah-like placeholder if needed, or default
                # ... add other IDs as needed or let backend handle names if supported
            }
            # If voice is in map, use ID, else let string pass through (ElevenLabs might handle it or error)
            # For robustness, we stick to the ones we know or pass the name if the API supports it. 
            # ElevenLabs API on Fal usually needs ID or name. 
            voice_id = voice_map.get(voice, voice) 

            arguments = {
                "text": prompt,
                "voice": voice_id,
                "speed": speed,
                "stability": stability,
                "similarity_boost": similarity_boost,
            }
            if style > 0:
                arguments["style"] = style
            if language_code:
                arguments["language_code"] = language_code

        elif engine == "sfx":
            model = "fal-ai/elevenlabs/sound-effects/v2"
            arguments = {
                "text": prompt,
                "prompt_influence": prompt_influence,
            }
            if duration_seconds:
                arguments["duration_seconds"] = min(duration_seconds, 22.0)

        elif engine == "music":
            model = "fal-ai/elevenlabs/music"
            arguments = {
                "prompt": prompt,
                "output_format": "mp3_44100_128",
                "force_instrumental": force_instrumental,
            }
            if duration_seconds:
                arguments["music_length_ms"] = int(min(duration_seconds, 600.0) * 1000)
        else:
            return "Invalid audio engine."

        if ctx:
            await ctx.info(f"Generating {engine} with '{model}'...")

        result = await fal_client.subscribe_async(
            model, arguments=arguments, with_logs=True, client_timeout=120.0
        )

        public_url = None
        for key in ["audio", "audio_file"]:
            if key in result and isinstance(result[key], dict):
                public_url = result[key].get("url")
                if public_url:
                    break

        if public_url:
            await download_file(public_url, output_path)
            archive = copy_to_archive(output_path)
            return format_result("audio", engine, str(output_path), public_url, str(archive))
        return f"Unexpected response from {engine}: {str(result)}"

    except Exception as e:
        return f"Failed to generate audio: {str(e)}"


# =============================================================================
# TALKING AVATAR
# =============================================================================

@mcp.tool()
async def create_talking_avatar(
    image_url: str,
    audio_url: str,
    output_filename: str,
    engine: Literal["live-avatar", "kling-avatar"] = "live-avatar",
    ctx: Context = None
) -> str:
    """
    Generates a talking avatar (lipsync) from image + audio. Returns LOCAL_PATH, PUBLIC_URL, and ARCHIVE_COPY.

    Call consult_multimedia_options("avatar") first to discuss the workflow with the user.

    Engines:
    - live-avatar: Natural real-time expressions.
    - kling-avatar: Kling AI Avatar v2, studio-grade lipsync.

    image_url and audio_url accept EITHER:
    - A PUBLIC_URL (https://...) from generate_image / generate_audio output
    - A local file path (will be auto-uploaded to Fal.ai)

    Recommended workflow:
    1. generate_image → get PUBLIC_URL for portrait
    2. generate_audio(engine="tts") → get PUBLIC_URL for speech
    3. create_talking_avatar(image_url=PUBLIC_URL, audio_url=PUBLIC_URL)
    """
    try:
        check_fal_key()
        output_path = resolve_output_path(output_filename)

        if ctx:
            await ctx.info("Resolving image and audio URLs...")
        image_url = await ensure_public_url(image_url)
        audio_url = await ensure_public_url(audio_url)

        if engine == "live-avatar":
            model = "fal-ai/live-avatar"
        elif engine == "kling-avatar":
            model = "fal-ai/kling-video/ai-avatar/v2/standard"
        else:
            return "Invalid avatar engine."

        if ctx:
            await ctx.info(f"Generating talking avatar with '{model}'... This may take 60-180 seconds.")
            await ctx.report_progress(progress=0.1, total=1.0)

        result = await fal_client.subscribe_async(
            model,
            arguments={"image_url": image_url, "audio_url": audio_url},
            with_logs=True,
            client_timeout=300.0
        )

        if ctx:
            await ctx.report_progress(progress=0.9, total=1.0)

        if "video" in result:
            public_url = result["video"]["url"]
            await download_file(public_url, output_path)
            archive = copy_to_archive(output_path)
            return format_result("talking avatar", engine, str(output_path), public_url, str(archive))
        return f"Unexpected response: {str(result)}"

    except Exception as e:
        return f"Failed to create talking avatar: {str(e)}"


if __name__ == "__main__":
    mcp.run()
