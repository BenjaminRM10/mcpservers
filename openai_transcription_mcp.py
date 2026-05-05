import asyncio
import base64
import json
import mimetypes
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import Context, FastMCP


mcp = FastMCP("openai-transcription")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_AUDIO_SUFFIXES = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}
JSON_LIKE_FORMATS = {"json", "verbose_json", "diarized_json"}
DOC_URLS = {
    "speech_to_text_guide": "https://developers.openai.com/api/docs/guides/speech-to-text",
    "transcriptions_api_reference": "https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create",
    "translations_api_reference": "https://developers.openai.com/api/reference/resources/audio/subresources/translations/methods/create",
    "gpt_4o_mini_transcribe_model": "https://developers.openai.com/api/docs/models/gpt-4o-mini-transcribe",
    "gpt_4o_transcribe_model": "https://developers.openai.com/api/docs/models/gpt-4o-transcribe",
    "gpt_4o_transcribe_diarize_model": "https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize",
    "whisper_model": "https://developers.openai.com/api/docs/models/whisper-1",
}


@dataclass
class PreparedInput:
    path: Path
    source: str
    cleanup: bool = False


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _current_date() -> str:
    return date.today().isoformat()


def _normalize_api_base() -> str:
    raw = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    ).rstrip("/")
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return raw + "/v1"
    return raw


def _get_headers() -> dict[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY no esta configurada. Agregala en el env del MCP."
        )

    headers = {"Authorization": f"Bearer {api_key}"}
    organization = os.getenv("OPENAI_ORGANIZATION")
    project = os.getenv("OPENAI_PROJECT")
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project
    return headers


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_data_url(value: str) -> bool:
    return value.startswith("data:")


def _guess_suffix(source: str, content_type: Optional[str] = None) -> str:
    suffix = Path(urlparse(source).path).suffix.lower()
    if suffix in SUPPORTED_AUDIO_SUFFIXES:
        return suffix

    if content_type:
        mime = content_type.split(";")[0].strip().lower()
        guessed = mimetypes.guess_extension(mime) or ""
        if guessed == ".oga":
            guessed = ".ogg"
        if guessed in SUPPORTED_AUDIO_SUFFIXES:
            return guessed

    return ".bin"


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def _write_temp_file(raw: bytes, suffix: str) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix="openai_transcription_", suffix=suffix)
    os.close(fd)
    path = Path(tmp_name)
    path.write_bytes(raw)
    return path


def _decode_data_url(data_url: str) -> PreparedInput:
    header, sep, payload = data_url.partition(",")
    if sep != "," or ";base64" not in header:
        raise ValueError("Data URL invalida. Se esperaba formato base64.")

    mime = header[5:].split(";")[0] or "application/octet-stream"
    suffix = _guess_suffix("", mime)
    raw = base64.b64decode(payload)
    path = _write_temp_file(raw, suffix)
    return PreparedInput(path=path, source="data_url", cleanup=True)


async def _download_to_temp(url: str) -> PreparedInput:
    timeout = httpx.Timeout(180.0, connect=20.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        suffix = _guess_suffix(url, response.headers.get("content-type"))
        path = _write_temp_file(response.content, suffix)
        return PreparedInput(path=path, source=url, cleanup=True)


async def _prepare_input(source: str) -> PreparedInput:
    if _is_data_url(source):
        return _decode_data_url(source)
    if _is_url(source):
        return await _download_to_temp(source)

    path = Path(source).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return PreparedInput(path=path, source=str(path), cleanup=False)


def _cleanup_prepared(prepared: PreparedInput) -> None:
    if prepared.cleanup:
        prepared.path.unlink(missing_ok=True)


def _validate_input_file(path: Path) -> list[str]:
    warnings: list[str] = []
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_SUFFIXES:
        warnings.append(
            "La extension no esta en la lista documentada por OpenAI "
            f"({suffix or 'sin extension'}). El archivo se enviara de todos modos."
        )

    size_bytes = path.stat().st_size
    if size_bytes > MAX_UPLOAD_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise ValueError(
            f"OpenAI documenta un limite actual de 25 MB por archivo. "
            f"El archivo pesa {size_mb:.2f} MB."
        )

    return warnings


def _model_family(model: str) -> Optional[str]:
    if model == "whisper-1":
        return "whisper"
    if model.startswith("gpt-4o-mini-transcribe"):
        return "gpt_4o_mini"
    if model == "gpt-4o-transcribe" or (
        model.startswith("gpt-4o-transcribe-") and "diarize" not in model
    ):
        return "gpt_4o"
    if model.startswith("gpt-4o-transcribe-diarize"):
        return "diarize"
    return None


def _default_transcription_response_format(model: str) -> str:
    family = _model_family(model)
    if family == "diarize":
        return "diarized_json"
    return "json"


def _ensure_temperature(temperature: Optional[float]) -> None:
    if temperature is None:
        return
    if temperature < 0 or temperature > 1:
        raise ValueError("temperature debe estar entre 0 y 1.")


def _normalize_timestamp_granularities(
    timestamp_granularities: Optional[list[Literal["word", "segment"]]],
) -> list[str]:
    normalized: list[str] = []
    for item in timestamp_granularities or []:
        if item not in {"word", "segment"}:
            raise ValueError(
                "timestamp_granularities solo acepta 'word' o 'segment'."
            )
        if item not in normalized:
            normalized.append(item)
    return normalized


def _resolve_output_path(output_path: str) -> Path:
    path = Path(output_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _validate_vad_config(
    prefix_padding_ms: Optional[int],
    silence_duration_ms: Optional[int],
    threshold: Optional[float],
) -> None:
    if prefix_padding_ms is not None and prefix_padding_ms < 0:
        raise ValueError("vad_prefix_padding_ms no puede ser negativo.")
    if silence_duration_ms is not None and silence_duration_ms < 0:
        raise ValueError("vad_silence_duration_ms no puede ser negativo.")
    if threshold is not None and (threshold < 0 or threshold > 1):
        raise ValueError("vad_threshold debe estar entre 0 y 1.")


async def _prepare_speaker_reference_data_urls(
    references: list[str],
) -> tuple[list[str], list[PreparedInput]]:
    prepared_inputs: list[PreparedInput] = []
    data_urls: list[str] = []

    for reference in references:
        if _is_data_url(reference):
            prepared = _decode_data_url(reference)
        else:
            prepared = await _prepare_input(reference)
        prepared_inputs.append(prepared)
        mime_type = _guess_mime_type(prepared.path)
        encoded = base64.b64encode(prepared.path.read_bytes()).decode("ascii")
        data_urls.append(f"data:{mime_type};base64,{encoded}")

    return data_urls, prepared_inputs


def _api_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    return f"OpenAI API devolvio {response.status_code}: {payload}"


def _append_form_value(form_data: list[tuple[str, str]], key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        form_data.append((key, "true" if value else "false"))
        return
    form_data.append((key, str(value)))


def _save_output(path: Path, payload: Any) -> None:
    if isinstance(payload, (dict, list)):
        path.write_text(_json(payload) + "\n", encoding="utf-8")
        return
    path.write_text(str(payload), encoding="utf-8")


async def _perform_standard_request(
    endpoint: str,
    input_file: PreparedInput,
    form_data: list[tuple[str, str]],
    response_format: str,
) -> Any:
    headers = _get_headers()
    url = f"{_normalize_api_base()}{endpoint}"
    timeout = httpx.Timeout(900.0, connect=20.0)

    file_content = input_file.path.read_bytes()
    multipart_fields: list[tuple[str, Any]] = [
        (key, (None, value)) for key, value in form_data
    ]
    multipart_fields.append(
        ("file", (input_file.path.name, file_content, _guess_mime_type(input_file.path)))
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, files=multipart_fields)

    if response.status_code >= 400:
        raise RuntimeError(_api_error_message(response))

    if response_format in JSON_LIKE_FORMATS:
        return response.json()
    return response.text


async def _perform_streaming_request(
    endpoint: str,
    input_file: PreparedInput,
    form_data: list[tuple[str, str]],
) -> dict[str, Any]:
    headers = _get_headers()
    url = f"{_normalize_api_base()}{endpoint}"
    timeout = httpx.Timeout(900.0, connect=20.0)
    events: list[dict[str, Any]] = []
    segments: list[Any] = []
    text_parts: list[str] = []
    final_text = ""

    def flush_event(
        current_name: Optional[str],
        data_lines: list[str],
    ) -> tuple[Optional[str], list[str]]:
        nonlocal final_text

        if not current_name and not data_lines:
            return None, []

        raw = "\n".join(data_lines).strip()
        if not raw:
            return None, []
        if raw == "[DONE]":
            return None, []

        try:
            payload = json.loads(raw)
        except Exception:
            payload = {"raw": raw}

        event_type = None
        if isinstance(payload, dict):
            event_type = payload.get("type") or current_name
        else:
            event_type = current_name or "unknown"

        if isinstance(payload, dict):
            if event_type == "transcript.text.delta":
                delta = payload.get("delta") or payload.get("text_delta") or ""
                if delta:
                    text_parts.append(delta)
            elif event_type == "transcript.text.done":
                done_text = payload.get("text")
                if isinstance(done_text, str):
                    final_text = done_text
            elif event_type == "transcript.text.segment":
                segment = payload.get("segment") if "segment" in payload else payload
                segments.append(segment)

        events.append({"event": event_type or "unknown", "data": payload})
        return None, []

    file_content = input_file.path.read_bytes()
    multipart_fields: list[tuple[str, Any]] = [
        (key, (None, value)) for key, value in form_data
    ]
    multipart_fields.append(
        ("file", (input_file.path.name, file_content, _guess_mime_type(input_file.path)))
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            files=multipart_fields,
        ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    error = httpx.Response(
                        response.status_code,
                        headers=response.headers,
                        content=body,
                        request=response.request,
                    )
                    raise RuntimeError(_api_error_message(error))

                current_name: Optional[str] = None
                data_lines: list[str] = []

                async for line in response.aiter_lines():
                    if not line:
                        current_name, data_lines = flush_event(current_name, data_lines)
                        continue
                    if line.startswith("event:"):
                        current_name = line.removeprefix("event:").strip()
                        continue
                    if line.startswith("data:"):
                        data_lines.append(line.removeprefix("data:").strip())

                flush_event(current_name, data_lines)

    if not final_text:
        final_text = "".join(text_parts)

    return {"text": final_text, "segments": segments, "events": events}


@mcp.tool()
async def list_openai_transcription_capabilities() -> str:
    """
    Returns the current OpenAI transcription model matrix and doc links.
    """
    return _json(
        {
            "checked_on": _current_date(),
            "server": "openai-transcription",
            "models": [
                {
                    "model": "whisper-1",
                    "type": "general_purpose",
                    "official_notes": [
                        "Modelo general de speech-to-text.",
                        "Tambien soporta translation a ingles.",
                        "timestamp_granularities solo esta documentado para whisper-1.",
                        "stream no esta soportado para whisper-1.",
                    ],
                    "transcriptions_response_formats": [
                        "json",
                        "text",
                        "srt",
                        "verbose_json",
                        "vtt",
                    ],
                    "translations_response_formats": [
                        "json",
                        "text",
                        "srt",
                        "verbose_json",
                        "vtt",
                    ],
                    "docs": [DOC_URLS["whisper_model"]],
                },
                {
                    "model": "gpt-4o-mini-transcribe",
                    "snapshots": [
                        "gpt-4o-mini-transcribe-2025-03-20",
                        "gpt-4o-mini-transcribe-2025-12-15",
                    ],
                    "type": "frontier_fast",
                    "official_notes": [
                        "Mejor accuracy que Whisper con menor latencia y costo que gpt-4o-transcribe.",
                        "Prompts y logprobs si estan documentados.",
                        "La guia documenta response_format json o text; la referencia del endpoint tambien menciona json-only. Este MCP usa json por defecto y permite text con advertencia.",
                    ],
                    "transcriptions_response_formats": ["json", "text"],
                    "docs": [
                        DOC_URLS["gpt_4o_mini_transcribe_model"],
                        DOC_URLS["transcriptions_api_reference"],
                    ],
                },
                {
                    "model": "gpt-4o-transcribe",
                    "type": "frontier_quality",
                    "official_notes": [
                        "Mayor calidad que la variante mini.",
                        "Prompts y logprobs si estan documentados.",
                        "La guia documenta response_format json o text; la referencia del endpoint tambien menciona json-only. Este MCP usa json por defecto y permite text con advertencia.",
                    ],
                    "transcriptions_response_formats": ["json", "text"],
                    "docs": [
                        DOC_URLS["gpt_4o_transcribe_model"],
                        DOC_URLS["transcriptions_api_reference"],
                    ],
                },
                {
                    "model": "gpt-4o-transcribe-diarize",
                    "type": "speaker_diarization",
                    "official_notes": [
                        "Agrega speaker labels.",
                        "response_format diarized_json devuelve segmentos con speaker/start/end.",
                        "No soporta prompt, logprobs ni timestamp_granularities.",
                        "OpenAI recomienda chunking_strategy='auto', y es requerido para audios >30s.",
                        "No esta documentado en Realtime API.",
                    ],
                    "transcriptions_response_formats": [
                        "json",
                        "text",
                        "diarized_json",
                    ],
                    "docs": [
                        DOC_URLS["gpt_4o_transcribe_diarize_model"],
                        DOC_URLS["transcriptions_api_reference"],
                    ],
                },
            ],
            "docs_consistency_notes": [
                "La guia Speech-to-Text documenta text para gpt-4o-transcribe y gpt-4o-mini-transcribe.",
                "La referencia del endpoint de transcriptions hoy tambien afirma json-only para esos dos modelos.",
                "Las paginas de modelos muestran el endpoint /audio/translations, pero la guia Speech-to-Text y la referencia de /audio/translations hoy dicen que translation solo esta disponible con whisper-1.",
            ],
            "docs": DOC_URLS,
        }
    )


@mcp.tool()
async def transcribe_audio(
    audio_path_or_url: str,
    model: str = "gpt-4o-mini-transcribe",
    response_format: Optional[
        Literal["json", "text", "srt", "verbose_json", "vtt", "diarized_json"]
    ] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    include_logprobs: bool = False,
    timestamp_granularities: Optional[list[Literal["word", "segment"]]] = None,
    stream: bool = False,
    chunking_strategy: Optional[Literal["auto", "server_vad"]] = None,
    vad_prefix_padding_ms: Optional[int] = None,
    vad_silence_duration_ms: Optional[int] = None,
    vad_threshold: Optional[float] = None,
    known_speaker_names: Optional[list[str]] = None,
    known_speaker_reference_paths_or_urls: Optional[list[str]] = None,
    output_path: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Transcribes audio with OpenAI's current transcription models.

    Supports:
    - whisper-1
    - gpt-4o-mini-transcribe
    - gpt-4o-mini-transcribe-2025-03-20
    - gpt-4o-mini-transcribe-2025-12-15
    - gpt-4o-transcribe
    - gpt-4o-transcribe-diarize
    """
    warnings: list[str] = []
    prepared_input: Optional[PreparedInput] = None
    prepared_refs: list[PreparedInput] = []

    try:
        if ctx:
            await ctx.info(f"Preparando audio para transcripcion con {model}...")

        family = _model_family(model)
        if family is None:
            warnings.append(
                "El modelo no coincide con la matriz oficial actual, pero se enviara igualmente."
            )

        normalized_response_format = response_format or _default_transcription_response_format(
            model
        )
        normalized_timestamps = _normalize_timestamp_granularities(
            timestamp_granularities
        )
        _ensure_temperature(temperature)
        _validate_vad_config(
            vad_prefix_padding_ms,
            vad_silence_duration_ms,
            vad_threshold,
        )

        if family == "whisper":
            if normalized_response_format not in {
                "json",
                "text",
                "srt",
                "verbose_json",
                "vtt",
            }:
                raise ValueError(
                    "whisper-1 solo usa json, text, srt, verbose_json o vtt."
                )
            if stream:
                warnings.append(
                    "OpenAI documenta que whisper-1 no soporta streaming. Se desactiva stream."
                )
                stream = False
            if include_logprobs:
                raise ValueError("include_logprobs no esta soportado con whisper-1.")
            if normalized_timestamps and normalized_response_format != "verbose_json":
                raise ValueError(
                    "timestamp_granularities requiere response_format='verbose_json'."
                )
            if chunking_strategy or any(
                value is not None
                for value in (
                    vad_prefix_padding_ms,
                    vad_silence_duration_ms,
                    vad_threshold,
                )
            ):
                raise ValueError(
                    "chunking_strategy manual no esta documentado para whisper-1."
                )
            if known_speaker_names or known_speaker_reference_paths_or_urls:
                raise ValueError(
                    "known_speaker_* solo aplica a gpt-4o-transcribe-diarize."
                )

        if family in {"gpt_4o", "gpt_4o_mini"}:
            if normalized_response_format not in {"json", "text"}:
                raise ValueError(
                    f"{model} solo usa json o text segun la guia oficial."
                )
            if normalized_timestamps:
                raise ValueError(
                    "timestamp_granularities solo esta documentado para whisper-1."
                )
            if include_logprobs and normalized_response_format != "json":
                raise ValueError(
                    "include_logprobs requiere response_format='json'."
                )
            if normalized_response_format == "text":
                warnings.append(
                    "La guia oficial documenta text para este modelo, pero la referencia del endpoint hoy tambien dice json-only. Si OpenAI rechaza text, usa json."
                )
            if chunking_strategy or any(
                value is not None
                for value in (
                    vad_prefix_padding_ms,
                    vad_silence_duration_ms,
                    vad_threshold,
                )
            ):
                raise ValueError(
                    "chunking_strategy manual solo aplica a gpt-4o-transcribe-diarize."
                )
            if known_speaker_names or known_speaker_reference_paths_or_urls:
                raise ValueError(
                    "known_speaker_* solo aplica a gpt-4o-transcribe-diarize."
                )

        if family == "diarize":
            if normalized_response_format not in {"json", "text", "diarized_json"}:
                raise ValueError(
                    "gpt-4o-transcribe-diarize solo usa json, text o diarized_json."
                )
            if prompt:
                raise ValueError(
                    "prompt no esta soportado con gpt-4o-transcribe-diarize."
                )
            if include_logprobs:
                raise ValueError(
                    "include_logprobs no esta soportado con gpt-4o-transcribe-diarize."
                )
            if normalized_timestamps:
                raise ValueError(
                    "timestamp_granularities no esta soportado con diarization."
                )
            if chunking_strategy is None:
                chunking_strategy = "auto"
                warnings.append(
                    "Se usa chunking_strategy='auto' para diarization, siguiendo la recomendacion de OpenAI."
                )
            if chunking_strategy == "auto" and any(
                value is not None
                for value in (
                    vad_prefix_padding_ms,
                    vad_silence_duration_ms,
                    vad_threshold,
                )
            ):
                raise ValueError(
                    "Los parametros VAD manuales solo aplican con chunking_strategy='server_vad'."
                )
            if chunking_strategy not in {"auto", "server_vad"}:
                raise ValueError(
                    "chunking_strategy debe ser 'auto' o 'server_vad'."
                )
            if known_speaker_names or known_speaker_reference_paths_or_urls:
                if not known_speaker_names or not known_speaker_reference_paths_or_urls:
                    raise ValueError(
                        "Debes enviar known_speaker_names y known_speaker_reference_paths_or_urls juntos."
                    )
                if len(known_speaker_names) != len(
                    known_speaker_reference_paths_or_urls
                ):
                    raise ValueError(
                        "known_speaker_names y known_speaker_reference_paths_or_urls deben tener la misma longitud."
                    )
                if len(known_speaker_names) > 4:
                    raise ValueError(
                        "OpenAI documenta hasta 4 speakers conocidos."
                    )

        if ctx and _is_url(audio_path_or_url):
            await ctx.info("Descargando audio remoto antes de enviarlo a OpenAI...")
        prepared_input = await _prepare_input(audio_path_or_url)
        warnings.extend(_validate_input_file(prepared_input.path))

        speaker_reference_data_urls: list[str] = []
        if known_speaker_reference_paths_or_urls:
            if ctx:
                await ctx.info("Convirtiendo referencias de speaker a data URLs...")
            warnings.append(
                "OpenAI documenta que cada known_speaker_reference debe durar entre 2 y 10 segundos; este MCP no valida esa duracion localmente."
            )
            speaker_reference_data_urls, prepared_refs = (
                await _prepare_speaker_reference_data_urls(
                    known_speaker_reference_paths_or_urls
                )
            )

        form_data: list[tuple[str, str]] = []
        _append_form_value(form_data, "model", model)
        _append_form_value(form_data, "response_format", normalized_response_format)
        _append_form_value(form_data, "language", language)
        _append_form_value(form_data, "prompt", prompt)
        _append_form_value(form_data, "temperature", temperature)
        if stream:
            _append_form_value(form_data, "stream", True)
        if include_logprobs:
            form_data.append(("include[]", "logprobs"))
        for granularity in normalized_timestamps:
            form_data.append(("timestamp_granularities[]", granularity))

        if family == "diarize":
            if chunking_strategy == "auto":
                _append_form_value(form_data, "chunking_strategy", "auto")
            elif chunking_strategy == "server_vad":
                form_data.append(("chunking_strategy[type]", "server_vad"))
                _append_form_value(
                    form_data,
                    "chunking_strategy[prefix_padding_ms]",
                    vad_prefix_padding_ms,
                )
                _append_form_value(
                    form_data,
                    "chunking_strategy[silence_duration_ms]",
                    vad_silence_duration_ms,
                )
                _append_form_value(
                    form_data,
                    "chunking_strategy[threshold]",
                    vad_threshold,
                )

            for name in known_speaker_names or []:
                form_data.append(("known_speaker_names[]", name))
            for data_url in speaker_reference_data_urls:
                form_data.append(("known_speaker_references[]", data_url))

        if ctx:
            await ctx.info(
                "Enviando solicitud de transcripcion al endpoint /v1/audio/transcriptions..."
            )

        if stream:
            if ctx:
                await ctx.info("Recibiendo eventos SSE de transcripcion...")
            payload = await _perform_streaming_request(
                "/audio/transcriptions",
                prepared_input,
                form_data,
            )
        else:
            payload = await _perform_standard_request(
                "/audio/transcriptions",
                prepared_input,
                form_data,
                normalized_response_format,
            )

        saved_to = None
        if output_path:
            output_file = _resolve_output_path(output_path)
            if ctx:
                await ctx.info(f"Guardando transcripcion en {output_file}...")
            _save_output(output_file, payload)
            saved_to = str(output_file)

        return _json(
            {
                "ok": True,
                "checked_on": _current_date(),
                "endpoint": "/v1/audio/transcriptions",
                "model": model,
                "response_format": normalized_response_format,
                "stream": stream,
                "input_source": prepared_input.source if prepared_input else audio_path_or_url,
                "input_file": str(prepared_input.path) if prepared_input else None,
                "saved_to": saved_to,
                "warnings": warnings,
                "docs": DOC_URLS,
                "result": payload,
            }
        )
    except Exception as exc:
        return _json(
            {
                "ok": False,
                "checked_on": _current_date(),
                "endpoint": "/v1/audio/transcriptions",
                "model": model,
                "error": str(exc),
                "warnings": warnings,
                "docs": DOC_URLS,
            }
        )
    finally:
        if prepared_input is not None:
            _cleanup_prepared(prepared_input)
        for prepared_ref in prepared_refs:
            _cleanup_prepared(prepared_ref)


# ---------------------------------------------------------------------------
# Helpers para transcripcion de audios largos (ffmpeg requerido)
# ---------------------------------------------------------------------------


def _check_ffmpeg_available() -> None:
    """Lanza RuntimeError si ffmpeg o ffprobe no estan instalados."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg y ffprobe son necesarios para esta herramienta. "
            "Instala con:  sudo apt install ffmpeg  o  brew install ffmpeg"
        )


async def _run_ffprobe_duration(path: Path) -> float:
    """Retorna la duracion del audio en segundos usando ffprobe."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe error: {stderr.decode()}")
    text = stdout.decode().strip()
    if not text:
        raise RuntimeError("ffprobe no devolvio duracion del archivo.")
    return float(text)


async def _run_ffmpeg_cmd(*args: str) -> None:
    """Ejecuta un comando ffmpeg; lanza RuntimeError si falla."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {stderr.decode()}")


async def _normalize_volume(input_path: Path, output_path: Path) -> None:
    """Normaliza el volumen con filtro loudnorm de ffmpeg (estandar EBU R128)."""
    await _run_ffmpeg_cmd(
        "-y", "-v", "warning",
        "-i", str(input_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_path),
    )


async def _split_audio_to_chunks(
    input_path: Path,
    output_dir: Path,
    chunk_seconds: int,
) -> list[tuple[Path, float]]:
    """
    Divide el audio en segmentos de chunk_seconds usando ffmpeg segment muxer.
    Retorna lista de (ruta_chunk, offset_segundos) ordenada.
    """
    suffix = input_path.suffix.lower() or ".mp3"
    pattern = str(output_dir / f"chunk_%04d{suffix}")
    await _run_ffmpeg_cmd(
        "-y", "-v", "warning",
        "-i", str(input_path),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        "-reset_timestamps", "1",
        pattern,
    )
    chunks = sorted(output_dir.glob(f"chunk_*{suffix}"))
    return [(chunk, float(i * chunk_seconds)) for i, chunk in enumerate(chunks)]


async def _transcribe_chunk_internal(
    chunk_path: Path,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    response_format: str,
    offset_seconds: float,
) -> dict[str, Any]:
    """
    Transcribe un chunk directamente via OpenAI API (sin pasar por el tool MCP).
    Ajusta los timestamps de los segmentos sumando offset_seconds.
    Retorna dict con: text, segments (con timestamps ajustados).
    """
    family = _model_family(model)

    form_data: list[tuple[str, str]] = []
    _append_form_value(form_data, "model", model)
    _append_form_value(form_data, "response_format", response_format)
    _append_form_value(form_data, "language", language)
    if family != "diarize" and prompt:
        _append_form_value(form_data, "prompt", prompt)
    if family == "diarize":
        _append_form_value(form_data, "chunking_strategy", "auto")

    prepared = PreparedInput(path=chunk_path, source=str(chunk_path), cleanup=False)
    raw = await _perform_standard_request(
        "/audio/transcriptions",
        prepared,
        form_data,
        response_format,
    )

    text = ""
    segments: list[Any] = []

    if response_format in {"diarized_json", "verbose_json"} and isinstance(raw, dict):
        text = raw.get("text", "")
        for seg in raw.get("segments", []):
            adj = dict(seg)
            if "start" in adj and adj["start"] is not None:
                adj["start"] = round(adj["start"] + offset_seconds, 3)
            if "end" in adj and adj["end"] is not None:
                adj["end"] = round(adj["end"] + offset_seconds, 3)
            segments.append(adj)
    elif isinstance(raw, dict):
        text = raw.get("text", "")
    else:
        text = str(raw)

    return {"offset_seconds": offset_seconds, "text": text, "segments": segments}


@mcp.tool()
async def transcribe_long_audio(
    audio_path_or_url: str,
    model: str = "gpt-4o-transcribe-diarize",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    chunk_minutes: int = 10,
    normalize_volume: bool = True,
    output_path: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Transcribe un audio largo dividiendolo automaticamente en chunks y
    transcribiendo todos en paralelo con asyncio.gather.

    Requiere: ffmpeg y ffprobe instalados en el servidor del MCP.

    Modelos soportados: whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe,
    gpt-4o-transcribe-diarize.

    Notas:
    - chunk_minutes: tamano de cada chunk (default 10). Reducir si exceden 25 MB.
    - normalize_volume: aplica loudnorm EBU R128 antes de transcribir (recomendado).
    - prompt: NO se envia a gpt-4o-transcribe-diarize (no lo soporta ese modelo).
    - Para diarized_json, los speaker labels (speaker_0, speaker_1...) son locales
      a cada chunk y pueden no ser consistentes entre chunks distintos.
    - Los timestamps de los segmentos estan ajustados a la posicion en el audio total.
    """
    _check_ffmpeg_available()

    warnings: list[str] = []
    workdir: Optional[Path] = None
    prepared_input: Optional[PreparedInput] = None

    try:
        if ctx:
            await ctx.info("Iniciando transcripcion larga...")

        family = _model_family(model)
        if family is None:
            warnings.append(
                "El modelo no coincide con la matriz oficial actual, pero se enviara igualmente."
            )

        if family == "diarize":
            response_format = "diarized_json"
        elif family in {"gpt_4o", "gpt_4o_mini"}:
            response_format = "json"
        else:
            response_format = "verbose_json"

        if ctx and _is_url(audio_path_or_url):
            await ctx.info("Descargando audio remoto...")
        prepared_input = await _prepare_input(audio_path_or_url)

        workdir = Path(tempfile.mkdtemp(prefix="openai_long_txn_"))
        chunks_dir = workdir / "chunks"
        chunks_dir.mkdir()

        source_path = prepared_input.path

        if normalize_volume:
            if ctx:
                await ctx.info("Normalizando volumen (loudnorm I=-16 LUFS)...")
            normalized_path = workdir / f"normalized{source_path.suffix}"
            try:
                await _normalize_volume(source_path, normalized_path)
                source_path = normalized_path
                warnings.append(
                    "Volumen normalizado con loudnorm (I=-16 LUFS, TP=-1.5 dBTP, LRA=11)."
                )
            except RuntimeError as exc:
                warnings.append(
                    f"Normalizacion de volumen fallo, usando audio original: {exc}"
                )

        duration_seconds = await _run_ffprobe_duration(source_path)
        chunk_seconds = chunk_minutes * 60
        num_chunks_est = max(1, int((duration_seconds + chunk_seconds - 1) // chunk_seconds))

        if ctx:
            await ctx.info(
                f"Duracion: {duration_seconds:.0f}s (~{duration_seconds / 60:.1f}min). "
                f"Dividiendo en ~{num_chunks_est} chunks de {chunk_minutes}min..."
            )

        chunk_list = await _split_audio_to_chunks(source_path, chunks_dir, chunk_seconds)

        oversized = [str(c) for c, _ in chunk_list if c.stat().st_size > MAX_UPLOAD_BYTES]
        if oversized:
            size_mb = Path(oversized[0]).stat().st_size / (1024 * 1024)
            raise ValueError(
                f"Al menos un chunk excede 25 MB ({size_mb:.1f} MB). "
                f"Reduce chunk_minutes (actualmente {chunk_minutes}) e intenta de nuevo."
            )

        if ctx:
            await ctx.info(
                f"Transcribiendo {len(chunk_list)} chunks con '{model}' en paralelo..."
            )

        tasks = [
            _transcribe_chunk_internal(
                chunk_path=chunk_path,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                offset_seconds=offset,
            )
            for chunk_path, offset in chunk_list
        ]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        combined_text_parts: list[str] = []
        combined_segments: list[Any] = []
        failed_chunks: list[int] = []
        chunk_details: list[dict[str, Any]] = []

        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                failed_chunks.append(i + 1)
                warnings.append(f"Chunk {i + 1} fallo: {result}")
                chunk_details.append({
                    "chunk": i + 1,
                    "offset_seconds": chunk_list[i][1],
                    "status": "error",
                    "error": str(result),
                })
            else:
                combined_text_parts.append(result["text"])
                combined_segments.extend(result["segments"])
                chunk_details.append({
                    "chunk": i + 1,
                    "offset_seconds": result["offset_seconds"],
                    "status": "ok",
                    "text_length": len(result["text"]),
                    "segments_count": len(result["segments"]),
                })

        if failed_chunks:
            warnings.append(
                f"Chunks con error: {failed_chunks}. "
                "El texto puede tener huecos en esos segmentos."
            )

        combined_text = " ".join(p.strip() for p in combined_text_parts if p.strip())

        output_payload: dict[str, Any] = {
            "text": combined_text,
            "segments": combined_segments,
            "total_duration_seconds": duration_seconds,
            "total_chunks": len(chunk_list),
            "failed_chunks": failed_chunks,
            "chunks": chunk_details,
        }

        saved_to = None
        if output_path:
            output_file = _resolve_output_path(output_path)
            if ctx:
                await ctx.info(f"Guardando en {output_file}...")
            _save_output(output_file, output_payload)
            saved_to = str(output_file)

        if ctx:
            word_count = len(combined_text.split())
            await ctx.info(f"Listo. {len(chunk_list)} chunks, {word_count} palabras.")

        return _json({
            "ok": True,
            "checked_on": _current_date(),
            "model": model,
            "response_format": response_format,
            "total_duration_seconds": duration_seconds,
            "total_chunks": len(chunk_list),
            "chunk_minutes": chunk_minutes,
            "normalize_volume": normalize_volume,
            "saved_to": saved_to,
            "warnings": warnings,
            "docs": DOC_URLS,
            "result": output_payload,
        })

    except Exception as exc:
        return _json({
            "ok": False,
            "checked_on": _current_date(),
            "model": model,
            "error": str(exc),
            "warnings": warnings,
            "docs": DOC_URLS,
        })
    finally:
        if prepared_input is not None:
            _cleanup_prepared(prepared_input)
        if workdir is not None and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


@mcp.tool()
async def translate_audio_to_english(
    audio_path_or_url: str,
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
    prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    output_path: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """
    Translates audio into English using OpenAI's current translation endpoint.

    Official docs currently state that only whisper-1 is available here.
    """
    warnings: list[str] = []
    prepared_input: Optional[PreparedInput] = None

    try:
        _ensure_temperature(temperature)

        if response_format not in {"json", "text", "srt", "verbose_json", "vtt"}:
            raise ValueError(
                "response_format debe ser json, text, srt, verbose_json o vtt."
            )

        warnings.append(
            "La documentacion oficial actual de /audio/translations dice que solo whisper-1 esta disponible en este endpoint."
        )
        if prompt:
            warnings.append(
                "La referencia oficial de /audio/translations indica que el prompt debe estar en ingles."
            )

        if ctx:
            await ctx.info("Preparando audio para traduccion a ingles...")

        if ctx and _is_url(audio_path_or_url):
            await ctx.info("Descargando audio remoto antes de enviarlo a OpenAI...")
        prepared_input = await _prepare_input(audio_path_or_url)
        warnings.extend(_validate_input_file(prepared_input.path))

        form_data: list[tuple[str, str]] = []
        _append_form_value(form_data, "model", "whisper-1")
        _append_form_value(form_data, "response_format", response_format)
        _append_form_value(form_data, "prompt", prompt)
        _append_form_value(form_data, "temperature", temperature)

        if ctx:
            await ctx.info(
                "Enviando solicitud de traduccion al endpoint /v1/audio/translations..."
            )

        payload = await _perform_standard_request(
            "/audio/translations",
            prepared_input,
            form_data,
            response_format,
        )

        saved_to = None
        if output_path:
            output_file = _resolve_output_path(output_path)
            if ctx:
                await ctx.info(f"Guardando traduccion en {output_file}...")
            _save_output(output_file, payload)
            saved_to = str(output_file)

        return _json(
            {
                "ok": True,
                "checked_on": _current_date(),
                "endpoint": "/v1/audio/translations",
                "model": "whisper-1",
                "response_format": response_format,
                "input_source": prepared_input.source if prepared_input else audio_path_or_url,
                "input_file": str(prepared_input.path) if prepared_input else None,
                "saved_to": saved_to,
                "warnings": warnings,
                "docs": DOC_URLS,
                "result": payload,
            }
        )
    except Exception as exc:
        return _json(
            {
                "ok": False,
                "checked_on": _current_date(),
                "endpoint": "/v1/audio/translations",
                "model": "whisper-1",
                "error": str(exc),
                "warnings": warnings,
                "docs": DOC_URLS,
            }
        )
    finally:
        if prepared_input is not None:
            _cleanup_prepared(prepared_input)


if __name__ == "__main__":
    mcp.run()
