# FCP AI Toolkit prototype Core ML models

These are developer-converted, experimental model assets for local testing. They are not official upstream Core ML releases, a production-quality model set, or an application release. The intended release tag is `v1.2-prototype-20260923.1`; an inventory entry does not mean an asset has been published.

| Variant | Tensor contract | Prototype limits |
|---|---|---|
| `realesrgan-x4plus-tile256-fp16-macos13` | Float32 RGB NCHW `[1,3,256,256]` → `[1,3,1024,1024]`; FP16 internal computation | Native 4× model. A final 2× result requires a separately validated resize adapter. Reviewed photographic appearance was accepted for testing; periodic colour patterns can change substantially. A tile shape does not authorize arbitrary image sizes or 4K. |
| `instructir-7d-flex16to3840-fp32-macos13` | Float32 RGB NCHW; each spatial axis 16…3840; separate `[1,256]` task vector; same-size raw output | Caller must zero-pad right/bottom to multiples of 16 and crop exactly back. The graph range does not enforce multiples or pixel count. The 720p path has bounded evidence; 1080p optimization-hint equivalence and full 4K capacity are not established. A 4K resource attempt was aborted; the range is not a support promise. |
| `rife425-tile512-fp32-macos13` | Float32 two-RGB-frame concatenation `[1,6,512,512]`, dynamic time `[1,1,1,1]` → RGB `[1,3,512,512]` | Fixed 512 context, not 720p/4K certification. The reviewed appearance is accepted for prototype testing; fast-motion/text and context failures remain. Scene-cut detection, rational-time placement and safe tiling are adapter responsibilities, not features of the tensor graph. |
| `fbcnn-color-blind-512-fp32-macos13` | Float32 RGB `[1,3,512,512]` → raw RGB and `[1,1]` blind `qf` | Experimental and **default off**. JPEG-trained; required video-detail quality tests failed. No general H.264/HEVC enhancement claim. Exactly 512×512 is established; no implicit resize or arbitrary-size tiling. `100*(1-qf)` is the model's JPEG quality estimate, not codec QP or confidence. |

Inputs use opaque, SDR encoded sRGB RGB code values in `[0,1]`. Raw outputs require finite/shape checks before any display clipping. Tensor shape, numerical precision and package deployment target are distinct from tested application capacity. The measured reference is an M1 Pro with 16 GB; base M1 performance is untested. No package claims HDR/alpha/RAW support, full-video/A/V correctness, all-resolution performance or a long-duration memory guarantee. Product availability must enforce only independently verified task/adapter capabilities. No alternative inference engine is included.

## Download layout and integrity

[Release assets](provenance/release-assets.json) gives the byte size, SHA-256 and destination path for each unchanged model file. Each package has three separately downloadable flat names:

- `<variant>--Manifest.json` → `Manifest.json`
- `<variant>--model.mlmodel` → `Data/com.apple.CoreML/model.mlmodel`
- `<variant>--weight.bin` → `Data/com.apple.CoreML/weights/weight.bin`

The InstructIR vector asset is separate from the package. An installer must validate every file before assembling the package, reject unsafe paths/symlinks, stage atomically and compile locally using Core ML. No Python, shell, archive utility, tokenizer or language model is needed in the product. Compiled caches are not distributed. This inventory is not the product capability/availability manifest.

## Provenance and licenses

[Model provenance](provenance/models.json), [upstream file inventory](provenance/upstream-files.json), [notices](NOTICE.md) and the original texts in [LICENSES](LICENSES) preserve source revisions, checkpoint identities and attribution. Real-ESRGAN x4plus and FBCNN weight coverage is explicitly a **project-license-scope inference**, not a separate weight-license grant. RIFE's official README expressly extends MIT to linked model contents. InstructIR's author-hosted model repository declares MIT; BGE's fixed-vector dependency is also MIT.

The package bytes are preserved exactly. Some embedded descriptions reflect the earlier conversion-time status, including RIFE's old redistribution-review wording and x4plus's old GPU-review wording. Those strings are historical, not an upstream license restriction; current distribution scope and caveats are documented here. No model metadata was rewritten to make a new hash appear qualified.

## Developer reproduction

[Conversion instructions](converters/README.md) use explicit checkpoint/output paths and the narrowly vendored official inference source. No private image, reference output, fixture hash, local test log, personal filesystem path, app implementation or credential is included. The portable recipes reconstruct the conversion operations; they were syntax/static-checked without rerunning conversion during staging. Compiler versions, trace inputs and serialization can affect emitted bytes, so regenerated packages require fresh hashes and numerical/native validation. The supplied release bytes are identified independently by the asset inventory.

## animevideov3 2× beta model

`v1.2-anime-20260924.1` adds the official Real-ESRGAN SRVGGNetCompact animevideov3 weights with the installed NCNN x2 graph semantics: internal 4×, bicubic half-pixel reduction to 2× before clipping. This is not a native 2× network. Float32 RGB NCHW input `[1,3,256,256]`, output `[1,3,512,512]`, FP16 internal computation. Reflected halo32, core192, crop-halo stitching; output is already 2× and must not be resized again. CPU+GPU Core ML inference remains local.

All 53 weight tensors were matched to the official checkpoint, accounting for NCNN FP16 truncation and denormal flushing; the conversion uses those exact stored convolution weights. See [provenance](provenance/animevideov3-x2.json) for checkpoint, source, package hashes and numerical parity. The model is specialized for anime/illustrated content and may smooth fine photographic texture. No universal quality improvement or long-video certification is claimed. Real-ESRGAN BSD-3-Clause covers the vendored SRVGG source; checkpoint redistribution follows the same project-license-scope inference as x4plus.

Reproduction: Python 3.12, torch 2.5.1, coremltools 8.1, numpy 1.26.4. Run `python converters/convert_anime_x2.py --checkpoint /path/to/realesr-animevideov3.pth --ncnn-weights /path/to/realesr-animevideov3-x2.bin --output /new/output/directory`. Rebuilt bytes may differ and require fresh integrity/numerical checks.

## Native 4× animevideov3 beta variant

`v1.2-anime4x-20260925.1` adds a separate native 4× package using the same NCNN-matched official weights, with no output reduction. Float32 RGB `[1,3,256,256]` to `[1,3,1024,1024]`, FP16 internals. Use `converters/convert_anime_scales.py --scale 4 --checkpoint /path/to/realesr-animevideov3.pth --ncnn-weights /path/to/realesr-animevideov3-x2.bin --output /new/output`. The x2 and x4 NCNN binaries have the same weight hash. The previous 2× release is unchanged. See [4× provenance](provenance/animevideov3-x4.json).

Use reflect halo32, core192 and crop-halo stitching, with no final resize. The app's 4K output boundary permits source long edge ≤960 and short edge ≤540. 8K video output is not qualified. Original Real-ESRGAN BSD-3-Clause notices and project-scope checkpoint coverage apply.
