# ONNX export parity fix — EEGConformer

## Symptom

`scripts/export_braindecode_eegconformer.py` produced a valid ONNX file but
the parity smoke-test failed:

```
PyTorch↔ORT cosine = 0.304700
assert cos > 0.999  # FAILED
```

PyTorch-side training and `evaluate.py` (which uses the same fc-hook trick)
worked correctly — only the exported ONNX `embedding` output diverged.

## Root cause

The export wrapper extracted the embedding via a forward hook on
`self.model.fc`, stored it on `self._embedding`, and returned it from
`forward`:

```python
target = self.model.fc
target.register_forward_hook(self._capture_embedding)
def forward(self, x):
    logits = self.model(x)         # hook sets self._embedding as side-effect
    return self._embedding, logits
```

`torch.onnx.export` traces tensor flow from the function's _return_ values.
The hook-captured tensor reached the return tuple only through a Python
attribute assignment (a side-effect outside the dataflow graph). Combined
with `do_constant_folding=True`, the exporter wired the named `embedding`
output to a folded/aliased intermediate node rather than to the actual
`fc` output. Result: ONNX `embedding` ≠ PyTorch `fc` output → cosine ≈ 0.30.

PyTorch eager execution (used by `training/scripts/evaluate.py`) is
unaffected because hooks fire deterministically on every real forward
pass; only the ONNX trace is brittle.

## Fix

Replace the hook with an explicit replication of
`braindecode.models.eegconformer.EEGConformer.forward`, returning both
tensors directly from the wrapper:

```python
def forward(self, x):
    x = torch.unsqueeze(x, dim=1)
    x = self.model.patch_embedding(x)
    feature = self.model.transformer(x)
    embedding = self.model.fc(feature)
    logits   = self.model.final_layer(embedding)
    return embedding, logits
```

This guarantees:

- both outputs are first-class tracer outputs (no side-effects),
- constant-folding cannot re-bind the `embedding` output name,
- module call order matches the upstream model exactly (verified against
  braindecode `EEGConformer.forward`, which does `unsqueeze → patch_embedding
→ transformer → fc → final_layer`).

## Validation

The existing parity assertion in the script is preserved unchanged
(`assert cos > 0.999`); the smoke-test re-runs both PyTorch and
onnxruntime on the same dummy tensor. Re-export against the trained
`eegconformer.pt` checkpoint should now print cosine ≈ 1.0.

## Files changed

- `scripts/export_braindecode_eegconformer.py` — wrapper rewritten;
  parity test untouched.
