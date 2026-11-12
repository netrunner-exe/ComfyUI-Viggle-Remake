# SPDX-License-Identifier: Apache-2.0
"""Viggle Remake on ComfyUI's built-in MiniMax H3 support.

Stock ComfyUI already runs the MiniMax-H3 transformer, both VAEs and the Qwen3-VL conditioner. These nodes add the
three things Viggle Remake does differently, each written to reproduce `inference/sample.py`:

RemakeConditioning   the source clip as <Video 1>, each image as <Picture N> and, with `pins`, also as a keyframe at
                     that real frame of the output; a text-only presentation; the source soundtrack held as the
                     clean target audio for the whole denoise
RemakeKeyframeUntil  drops the keyframes once sigma falls below `keyframe_until` (0.9 = the last two of six forwards)
RemakeNoise          draws the starting noise in the reference script's order, so a seed gives the same video
RemakePrefix         a hop: the previous take's frames open this render, held clean like the soundtrack

`width`, `height` and `length` of 0 mean the source's own value. The transformer
patchifies the latents in 2x2 blocks, so every image side is rounded to a multiple of 32 (with a warning in the log).
The reference script used 17n+5 frames; other lengths are accepted with a warning.

Load the LoRA with the stock LoraLoaderBypassModelOnly. LoraLoaderModelOnly merges it into the bf16 weights,
where rounding drops part of the update and the output comes out softer.
"""

import logging
import re
from collections.abc import Mapping

import torch

import comfy.audio
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.utils
import node_helpers

LOG = logging.getLogger("ViggleRemake")
FPS, AUDIO_SR, AUDIO_LATENT_FPS = 24, 32000, 40
UNBOUNDED = 0xffffffffffffffff


def lanczos(image, width, height):
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)


def stamps(nf):
    # the conditioner's 2 fps block labels for an nf-frame clip: every 12th frame, merged in pairs, each pair
    # labelled with its mean time ("%.1f" rounds half to even, so the first block is <0.2 seconds>)
    idx, cursor = [], 0.0
    while round(cursor) < nf:
        if not idx or round(cursor) > idx[-1]:
            idx.append(round(cursor))
        cursor += FPS / 2
    t = [i / 2 for i in range(len(idx))]
    t += t[-1:] * (len(t) % 2)
    return [(t[i] + t[i + 1]) / 2 for i in range(0, len(t), 2)]


def round32(size):
    return max(32, round(size / 32) * 32)


def aligned(name, width, height):
    # the transformer patchifies the latents in 2x2 blocks of a 16x VAE: both sides must be multiples of 32
    aligned_width, aligned_height = round32(width), round32(height)
    if (aligned_width, aligned_height) != (width, height):
        LOG.warning("%s is %dx%d, not a multiple of 32: resized to %dx%d", name, width, height,
                    aligned_width, aligned_height)
    return aligned_width, aligned_height


def canvas_size(source, width, height):
    # a side of 0 keeps the source's own size
    return aligned("canvas", width if width > 0 else source.shape[2], height if height > 0 else source.shape[1])


def fit_length(source, length):
    # 0 keeps the source's own frame count; a smaller length trims it, a larger one is clamped to it
    nf = source.shape[0]
    if not length or length <= 0:
        return source
    if length > nf:
        LOG.warning("length %d exceeds the source's %d frames: %d frames are used", length, nf, nf)
    return source[:min(length, nf)]


def fit_canvas(source, width, height):
    # scale to cover the canvas, then centre-crop
    if (source.shape[2], source.shape[1]) == (width, height):
        return source
    return comfy.utils.common_upscale(source.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)


def check_frame_count(nf):
    if (nf - 5) % 17:
        LOG.warning("source has %d frames, not 17n+5: the reference script used 17n+5, "
                    "the result is not verified for other lengths", nf)


def parse_pins(text, ref_count, nf):
    # whitespace, commas and semicolons separate the frame indices
    tokens = [t for t in re.split(r"[,;\s]+", (text or "").strip()) if t]
    pins = []
    for token in tokens:
        try:
            pins.append(int(token))
        except ValueError:
            raise ValueError(f"pins: '{token}' is not an integer frame index (got '{text}')") from None
    if len(pins) not in (0, ref_count):
        raise ValueError(f"pins: give one frame per reference image or none, got {len(pins)} pins for {ref_count} references")
    bad = [p for p in pins if not 0 <= p < nf]
    if bad:
        raise ValueError(f"pins: {bad} outside the output frames [0, {nf})")
    return pins


def read_audio(audio):
    # (waveform, sample_rate) of an AUDIO-like mapping with a non-empty waveform, else None. Any Mapping is accepted:
    # VideoHelperSuite returns a lazy mapping, not a dict, and it is read once to avoid decoding the track twice
    if not isinstance(audio, Mapping):
        return None
    try:
        waveform, sample_rate = audio["waveform"], audio["sample_rate"]
    except KeyError:
        return None
    filled = torch.is_tensor(waveform) and waveform.ndim == 3 and waveform.shape[1] > 0 and waveform.shape[2] > 0
    return (waveform, sample_rate) if filled and sample_rate else None


def warn_audio_length(duration, nf):
    video = nf / FPS
    if abs(duration - video) > 1 / FPS:
        LOG.warning("audio is %.3f s, video is %.3f s: the audio is %s", duration, video,
                    "trimmed" if duration > video else "padded with silence")


def target_audio(audio, nf):
    # a stereo waveform at AUDIO_SR of exactly (nf + 1) / FPS seconds: nf / FPS of the soundtrack, then silence
    n = round((nf + 1) / FPS * AUDIO_SR)
    wav = torch.zeros(2, n)
    loaded = read_audio(audio)
    if loaded is not None:
        wav, sr = loaded[0][0].float(), loaded[1]
        warn_audio_length(wav.shape[1] / sr, nf)
        wav = wav.repeat(2, 1) if wav.shape[0] == 1 else wav[:2]
        wav = wav[:, :round(nf / FPS * sr)]
        wav = comfy.audio.resample(wav, sr, AUDIO_SR) if sr != AUDIO_SR else wav
    elif audio is not None:
        LOG.warning("the audio input (%s) is empty or not an AUDIO value: silence is used", type(audio).__name__)
    return torch.nn.functional.pad(wav, (0, max(0, n - wav.shape[1])))[:, :n]


def encode_audio(audio_vae, wav, nf):
    # zero-pad to whole 800-sample latent frames ourselves: VAE.encode would centre-crop instead, shifting the
    # track by up to 400 samples, which moves the latents as far as a different clip would
    wav = torch.nn.functional.pad(wav, (0, -wav.shape[1] % 800))
    return audio_vae.encode(wav[None].movedim(1, -1))[..., :round(nf / FPS * AUDIO_LATENT_FPS)]


def reference_blocks(vae, z, refs):
    # the source, then the images at their own size, each side rounded to a multiple of 32
    blocks = [{"kind": "video", "latent_t": z.shape[2], "latent_h": z.shape[-2], "latent_w": z.shape[-1],
               "ref_audio_t": 0, "latent": z, "audio_latent": None}]
    for i, r in enumerate(refs):
        width, height = aligned("reference %d" % (i + 1), r.shape[2], r.shape[1])
        if (r.shape[2], r.shape[1]) != (width, height):
            r = lanczos(r, width, height)
        latent = vae.encode(r)
        blocks.append({"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1], "latent": latent})
    return blocks


def keyframe_blocks(vae, refs, pins, width, height):
    # keyframes live on the output canvas: each is resized to it when its size differs
    keyframes = []
    for r, p in zip(refs, pins):
        if (r.shape[2], r.shape[1]) != (width, height):
            r = lanczos(r, width, height)
        keyframes.append({"resolved_frame_index": p, "latent": vae.encode(r)})
    return keyframes


class RemakeConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "source": ("IMAGE", {"tooltip": "The source clip's frames, at 24 fps. Supplies motion, camera, framing "
                                                "and everything the prompt does not change. Its frame count and size "
                                                "define the output when width, height or length is 0."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "The six-section prompt, used verbatim."}),
                "width": ("INT", {"default": 832, "min": 0, "max": UNBOUNDED, "step": 1,
                                  "tooltip": "Output width in pixels. 0: the source's width. Rounded to a multiple of 32."}),
                "height": ("INT", {"default": 480, "min": 0, "max": UNBOUNDED, "step": 1,
                                   "tooltip": "Output height in pixels. 0: the source's height. Rounded to a multiple of 32."}),
            },
            "optional": {
                "length": ("INT", {"default": 124, "min": 0, "max": UNBOUNDED, "step": 1,
                                   "tooltip": "Output length in frames at 24 fps. 0: the source's frame count. "
                                              "A smaller value trims the source, a larger one is clamped to it."}),
                "pins": ("STRING", {"default": "", "tooltip": "One output frame per reference image, e.g. '12, 190'. "
                                                              "Each reference is then pinned as a keyframe at that "
                                                              "frame for the first part of the denoise, resized to "
                                                              "the output size when it differs. Empty: no keyframes."}),
                "audio": ("AUDIO", {"tooltip": "The soundtrack kept in the output, usually the source's own. "
                                               "Silence when not connected or empty."}),
                "ref_1": ("IMAGE", {"tooltip": "<Picture 1>: usually a frame of the source painted in an image editor."}),
                "ref_2": ("IMAGE",),
                "ref_3": ("IMAGE",),
                "ref_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, clip, vae, audio_vae, source, prompt, width, height, length=124, pins="",
            audio=None, ref_1=None, ref_2=None, ref_3=None, ref_4=None):
        refs = [r[:1, ..., :3] for r in (ref_1, ref_2, ref_3, ref_4) if r is not None]
        source = source[..., :3]

        # length and canvas are handled separately: each falls back to the source when 0
        width, height = canvas_size(source, width, height)
        source = fit_canvas(fit_length(source, length), width, height)
        nf = source.shape[0]
        pins = parse_pins(pins, len(refs), nf)
        check_frame_count(nf)

        # text only: one label per 2 fps block of the source, one per reference, then the prompt, each tokenized on its own
        segments = ["<Video 1>: ", *["<%.1f seconds>" % s for s in stamps(nf)],
                    *["<Picture %d>: " % (i + 1) for i in range(len(refs))], prompt]
        tokens = {"qwen3vl_32b": [[t for s in segments for t in clip.tokenize(s)["qwen3vl_32b"][0]]]}
        positive = clip.encode_from_tokens_scheduled(tokens)

        z = vae.encode(source)
        blocks = reference_blocks(vae, z, refs)
        keyframes = keyframe_blocks(vae, refs, pins, width, height)
        values = {"minimax_refs": blocks, "minimax_keyframes": keyframes} if keyframes else {"minimax_refs": blocks}
        positive = node_helpers.conditioning_set_values(positive, values)

        za = encode_audio(audio_vae, target_audio(audio, nf), nf)

        # the target: video to generate (shape taken from the encoded source), audio held clean (noise_mask 0)
        video = torch.zeros(1, 24, z.shape[2], z.shape[-2], z.shape[-1])
        latent = {"samples": comfy.nested_tensor.NestedTensor((video, za)),
                  "noise_mask": comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(za)))}
        return (positive, latent)


class RemakeKeyframeUntil:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "keyframe_until": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "The keyframes condition every forward whose sigma is at least this; "
                                                    "below it they are dropped and the references alone remain."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, model, keyframe_until):
        def release(executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
            p = minimax_payload or {}
            if p.get("keyframes") and float(timestep.flatten()[0]) / 1000 < keyframe_until:
                refs = p.get("refs") or []
                minimax_payload = dict(p, keyframes=None, layout=None,
                                       cond_video_latents=[r["latent"] for r in refs if "latent" in r],
                                       cond_audio_latents=[r["audio_latent"] for r in refs if r.get("audio_latent") is not None])
            return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "viggle_remake_keyframe_until", release)
        return (m,)


class Noise:
    def __init__(self, seed, skip):
        self.seed, self.skip = seed, skip

    def generate_noise(self, input_latent):
        g = torch.Generator().manual_seed(self.seed)
        for shape in self.skip:
            torch.randn(shape, generator=g)
        return comfy.nested_tensor.NestedTensor([torch.randn(t.shape, generator=g) for t in input_latent["samples"].unbind()])


class RemakeNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
        }}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, positive, seed):
        # one CPU generator, drawn once per reference and keyframe latent before the target's noise
        c = positive[0][1]
        skip = [b["latent"].shape for b in c.get("minimax_refs", []) + c.get("minimax_keyframes", []) if "latent" in b]
        return (Noise(seed, skip),)


class RemakePrefix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "latent": ("LATENT",),
            "prefix": ("IMAGE", {"tooltip": "The previous take's frames, on this render's canvas."}),
            "prefix_frames": ("INT", {"default": 17, "min": 17, "max": 323, "step": 17,
                                      "tooltip": "How many of its frames open this render, a multiple of 17."}),
            "prefix_at": ("INT", {"default": -1, "min": -1, "max": 100000,
                                  "tooltip": "The previous take's frame they start at; -1: its last prefix_frames."}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Viggle Remake"

    def run(self, vae, latent, prefix, prefix_frames, prefix_at):
        # frames [at, at + k) of the previous take are this render's first k frames: their latents go in clean with
        # noise_mask 0, so like the held soundtrack they enter at the condition timestep, are never stepped and come
        # out unchanged (the reference worker's prefix rows)
        k = prefix_frames
        at = len(prefix) - k if prefix_at < 0 else prefix_at
        px = prefix[at:at + k, ..., :3]
        video, audio = latent["samples"].unbind()
        vmask, amask = latent["noise_mask"].unbind()
        n = k // 17 * 5
        assert k % 17 == 0 and n < video.shape[2], f"prefix_frames must be 17m and shorter than the render, got {k}"
        assert tuple(px.shape[:3]) == (k, video.shape[3] * 16, video.shape[4] * 16), \
            f"the prefix must be {k} frames on the {video.shape[4] * 16}x{video.shape[3] * 16} canvas, got {tuple(px.shape)}"
        # the VAE encodes 17-frame clips and drops the last 3 latents: pad to 17m + 5, keep the first 5m
        clean = vae.encode(torch.cat([px, px[-1:].repeat(5, 1, 1, 1)]))[:, :, :n]
        video, vmask = video.clone(), vmask.clone()
        video[:, :, :n] = clean.to(video)
        vmask[:, :, :n] = 0
        return ({**latent, "samples": comfy.nested_tensor.NestedTensor((video, audio)),
                 "noise_mask": comfy.nested_tensor.NestedTensor((vmask, amask))},)


NODE_CLASS_MAPPINGS = {"RemakeConditioning": RemakeConditioning, "RemakeKeyframeUntil": RemakeKeyframeUntil,
                       "RemakeNoise": RemakeNoise, "RemakePrefix": RemakePrefix}
NODE_DISPLAY_NAME_MAPPINGS = {"RemakeConditioning": "Viggle Remake Conditioning",
                              "RemakeKeyframeUntil": "Viggle Remake Keyframe Until",
                              "RemakeNoise": "Viggle Remake Noise",
                              "RemakePrefix": "Viggle Remake Prefix"}
