# Portable developer recipes

The release bytes were produced with Python 3.12.13, Torch 2.5.1, coremltools 8.1 and NumPy 1.26.4 on Apple Silicon. The converter warns that Torch 2.5.1 exceeds coremltools 8.1's tested Torch 2.4 version. Preserve that warning; successful conversion does not waive numerical checks.

These newly authored portable wrappers preserve the used upstream architecture, strict loading and conversion adaptations. Their original local recipe names and SHA-256 are recorded in `provenance/models.json`; the original harnesses also performed private local qualification and are not redistributed. The wrappers here were syntax/static-checked only, without model import or conversion during public staging. Rebuilding requires fresh parity checks, native loading/GPU verification and resource/capacity qualification. Package serialization is not guaranteed bit-identical to the supplied release assets.

## Model conversion

Acquire only the exact official checkpoint referenced in `provenance/models.json`. RIFE's checkpoint must be extracted from the stated archive member. Supply explicit local file paths; the script verifies the expected checkpoint and vendored source hashes. Without `--execute`, it checks identities and exits without importing the model libraries.

```sh
python converters/convert.py x4plus --checkpoint INPUT/RealESRGAN_x4plus.pth --output OUTPUT/x4plus.mlpackage
python converters/convert.py instructir --checkpoint INPUT/im_instructir-7d.pt --output OUTPUT/instructir.mlpackage
python converters/convert.py rife --checkpoint INPUT/flownet.pkl --output OUTPUT/rife.mlpackage
python converters/convert.py fbcnn --checkpoint INPUT/fbcnn_color.pth --output OUTPUT/fbcnn.mlpackage
```

Append `--execute` only when a bounded developer conversion is intended. Use a fresh output directory and an external process/resource limit appropriate to the machine. The wrapper uses two CPU threads but that is not a memory guard. No four-model batch or 4K reference pass is implied.

The default trace sample is seed-1202 generated RGB noise at each variant's trace shape. `--sample-npy` may instead name a caller-owned float32 NCHW NumPy file of the exact documented shape. For InstructIR that is `[1,3,720,1280]`; no hidden resizing or image decoding is performed. The original flexible and FBCNN traces used locally retained test images; those images and their identities are intentionally absent. Their graph operations and weights, not their private qualification fixtures, are the reproducibility target. No new result may inherit the old package's hash or validation status.

The converter uses `skip_model_load=True` to avoid an implicit native model load during recipe execution. It does not run Core ML predictions, archive or publish output, silently download files, or overwrite an existing package. Original package metadata is restored from the public provenance record; historical descriptions remain historical.

## Fixed InstructIR vectors

The supplied `provenance/fixed-task-embeddings.json` is byte-identical to the runtime vector asset. The model expects one exact `[1,256]` Float32 vector per task. It has no runtime tokenizer or prompt encoder.

To regenerate vectors offline, acquire the frozen files listed in `provenance/instructir-upstream-inputs.json` into the listed relative names beneath an input directory, then:

```sh
python converters/prepare_vectors.py --inputs-dir INPUT --output OUTPUT/fixed-task-embeddings.json
```

Append `--execute` to run the CPU text model/head after all hashes match. The output must match both published raw float32 vector hashes. No images are involved. Input metadata declares the author's image/head revision and the BGE revision; original MIT licenses are included.

## Original changes

See `NOTICE.md` for exact source adaptations. All files in `vendor/` are unchanged official files; edits are explicit in the wrappers. The source inventory binds the vendored bytes to official revisions or the exact RIFE archive. Keep notices and changed-file disclosures when redistributing derivatives. Upstream model authors do not endorse this conversion.
