import os
import asyncio
import base64
from pathlib import Path
from typing import Optional, List
from mcp.server.fastmcp import FastMCP
from google import genai
from google.genai import types
from PIL import Image
import io

# Initialize FastMCP server
mcp = FastMCP("google-image-gen")

def get_client():
    """Helper to get the GenAI client with API key from environment."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)

@mcp.tool()
async def generate_image(
    prompt: str,
    output_filename: str,
    model: str = "gemini-2.5-flash-image",
    aspect_ratio: str = "1:1",
    number_of_images: int = 1,
    resolution: str = "1K",
    reference_image_paths: List[str] = [],
    use_grounding: bool = False
) -> str:
    """
    Generate an image using Google's Gemini models (Nano Banana).
    Supports standard generation, reference images (Gemini 3 Pro), and grounding.

    Args:
        prompt: Description of the image.
        output_filename: Path to save the image.
        model: "gemini-2.5-flash-image" (Nano Banana) or "gemini-3-pro-image-preview" (Nano Banana Pro).
        aspect_ratio: "1:1", "16:9", "4:3", etc.
        number_of_images: Number of images to generate (saves just the first one usually).
        resolution: "1K", "2K", "4K" (Only for Gemini 3 Pro).
        reference_image_paths: List of paths to local images to use as references (Gemini 3 Pro only).
        use_grounding: If True, uses Google Search to ground the image (Gemini 3 Pro only).
    """
    try:
        client = get_client()

        # Build configuration
        image_config_args = {"aspect_ratio": aspect_ratio}
        
        # Resolution is only for Pro models (or specifically supported ones)
        if "pro" in model.lower():
            image_config_args["image_size"] = resolution

        config_args = {
            "response_modalities": ["TEXT", "IMAGE"],
            "image_config": types.ImageConfig(**image_config_args)
        }

        # Add grounding tool if requested
        if use_grounding:
            config_args["tools"] = [{"google_search": {}}]

        config = types.GenerateContentConfig(**config_args)
        
        # Prepare contents
        contents = [prompt]
        
        # Add reference images if provided
        for ref_path in reference_image_paths:
            try:
                img = Image.open(ref_path)
                contents.append(img)
            except Exception as e:
                return f"Error loading reference image '{ref_path}': {e}"

        print(f"Generating image with model '{model}'...")
        
        # Choose method based on model type (Flash Image usually uses generate_images, Pro uses generate_content)
        # However, the prompt examples show generate_content for everything except the very first simple example
        # but even that one shows generate_content in some contexts. 
        # Actually, the first example for Nano Banana uses `client.models.generate_content`.
        # The OLD API used `generate_images`, the NEW one seems to unify under `generate_content`.
        # Let's use `generate_content` as per valid examples in prompt.

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

        return save_response_images(response, output_filename)

    except Exception as e:
        return f"Failed to generate image: {str(e)}"

@mcp.tool()
async def edit_image(
    prompt: str,
    base_image_path: str,
    output_filename: str,
    model: str = "gemini-2.5-flash-image",
) -> str:
    """
    Edit an existing image based on a text prompt (Inpainting/Editing).

    Args:
        prompt: Instruction for the edit (e.g., "Add a hat to the cat").
        base_image_path: Path to the local image to edit.
        output_filename: Path to save the result.
        model: "gemini-2.5-flash-image" or "gemini-3-pro-image-preview".
    """
    try:
        client = get_client()
        
        if not os.path.exists(base_image_path):
            return f"Error: Base image not found at {base_image_path}"
            
        base_image = Image.open(base_image_path)
        
        # Config (usually doesn't need much for editing, but good to have)
        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        )

        print(f"Editing image with model '{model}'...")
        
        response = client.models.generate_content(
            model=model,
            contents=[prompt, base_image],
            config=config
        )

        return save_response_images(response, output_filename)

    except Exception as e:
        return f"Failed to edit image: {str(e)}"

def save_response_images(response, output_filename_base) -> str:
    """Helper to parse response and save images."""
    saved_files = []
    
    if not response.parts:
        return "No content returned from model."

    output_path = Path(output_filename_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    image_count = 0
    
    for part in response.parts:
        # Check for inline_data (image)
        if part.inline_data:
            image_count += 1
            # If multiple images, append number
            if image_count > 1:
                final_path = output_path.with_name(f"{output_path.stem}_{image_count}{output_path.suffix}")
            else:
                final_path = output_path

            try:
                # Part.as_image() returns a PIL Image
                # But sometimes it might not be available directly if we didn't import PIL in the right scope?
                # The SDK usually returns a PIL Image object from .as_image()
                img = part.as_image()
                img.save(final_path)
                saved_files.append(str(final_path))
            except Exception as e:
                # Fallback if as_image() fails, try decoding base64 manually
                try:
                    image_data = base64.b64decode(part.inline_data.data)
                    with open(final_path, "wb") as f:
                        f.write(image_data)
                    saved_files.append(str(final_path))
                except Exception as inner_e:
                     return f"Failed to save image part: {e}, {inner_e}"

    if not saved_files:
        # Check if it was just text (refusal or error)
        texts = [p.text for p in response.parts if p.text]
        return f"No images generated. Model output: {' '.join(texts)}"

    return f"Successfully generated/edited images: {', '.join(saved_files)}"

@mcp.tool()
async def enhance_prompt(prompt: str) -> str:
    """
    Enhances a simple prompt into a detailed image generation prompt using Gemini 2.0 Flash.

    Args:
        prompt: The basic concept or simple prompt.

    Returns:
        A detailed, descriptive prompt suitable for image generation models.
    """
    try:
        client = get_client()
        enhancer_model = "gemini-2.0-flash" 

        system_instruction = (
            "You are an expert at writing prompts for AI image generators. "
            "Take the user's input and expand it into a detailed, descriptive prompt "
            "focusing on lighting, composition, style, and texture. "
            "Keep it under 100 words. Return ONLY the enhanced prompt."
        )

        response = client.models.generate_content(
            model=enhancer_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        
        return response.text.strip()

    except Exception as e:
        return f"Failed to enhance prompt: {str(e)}"

if __name__ == "__main__":
    mcp.run()
