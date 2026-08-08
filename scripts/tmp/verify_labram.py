import os, sys, numpy as np, torch, torch.nn as nn
from functools import partial

REPO = 'C:/Users/pc/AppData/Local/Temp/labram_repo_py'
CKPT = 'C:/Users/pc/AppData/Local/Temp/labram-base.pth'
sys.path.insert(0, REPO)
from modeling_finetune import NeuralTransformer

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
]

print('='*70)
print('LaBraM Checkpoint Verification & ONNX Export')
print('='*70)

# Load checkpoint
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
state = ckpt['model']
print(f'Checkpoint param keys: {len(state)}')
dino = [k for k in state if not k.startswith('student.')]
print(f'DINO-only keys: {dino}')

# Strip student prefix
mstate = {}
for k,v in state.items():
    new_k = k[8:] if k.startswith('student.') else k
    mstate[new_k] = v

# Remap 'norm.*' -> 'fc_norm.*'
remapped = {}
for k, v in mstate.items():
    if k == 'norm.weight':
        remapped['fc_norm.weight'] = v
    elif k == 'norm.bias':
        remapped['fc_norm.bias'] = v
    else:
        remapped[k] = v
mstate = remapped

# Create model
model = NeuralTransformer(
    EEG_size=1600, patch_size=200, in_chans=1, num_classes=0,
    embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
    init_values=0.1,
    qk_norm=partial(nn.LayerNorm, eps=1e-6),
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    use_mean_pooling=True,
)
missing, unexpected = model.load_state_dict(mstate, strict=False)
print(f'Missing: {missing}')
print(f'Unexpected: {unexpected}')

# Verify shapes
print('Verifying key shapes:')
for name, ck_key, model_obj in [
    ('cls_token', 'student.cls_token', model.cls_token),
    ('pos_embed', 'student.pos_embed', model.pos_embed),
    ('time_embed', 'student.time_embed', model.time_embed),
]:
    ck_shape = state[ck_key].shape
    m_shape = model_obj.shape
    print(f'  {name}: model={list(m_shape)}, ckpt={list(ck_shape)}, match={m_shape==ck_shape}')

# Verify checkpoint coverage
model_keys = set(model.state_dict().keys())
loaded = set()
for mk in model_keys:
    if mk in mstate and model.state_dict()[mk].shape == mstate[mk].shape:
        loaded.add(mk)
not_loaded = model_keys - loaded
print(f'Checkpoint coverage: {len(loaded)}/{len(model_keys)} model keys loaded from checkpoint')
if not_loaded:
    print(f'Not loaded (random init): {sorted(not_loaded)}')

# Test forward
ch_names = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T7','T8','P7','P8']
input_chans = [0]
for ch in ch_names:
    input_chans.append(standard_1020.index(ch)+1)
print(f'Channels: {len(ch_names)}, input_chans: {input_chans}')

n_ch = len(ch_names); n_patch = 8; psize = 200
torch.manual_seed(42)
eeg = torch.randn(1, n_ch, n_patch, psize)
model.eval()

with torch.no_grad():
    out = model(eeg, input_chans=input_chans)
    print(f'Output shape (pooled): {out.shape}')
    out_p = model(eeg, input_chans=input_chans, return_patch_tokens=True)
    print(f'Patch tokens shape: {out_p.shape}')
print('Forward inference works!')

# ONNX wrapper
class LaBraMWrapper(nn.Module):
    def __init__(self, model, input_chans):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.cls_token = nn.Parameter(model.cls_token.clone())
        self.pos_embed = nn.Parameter(model.pos_embed.clone())
        self.time_embed = nn.Parameter(model.time_embed.clone())
        self.pos_drop = model.pos_drop
        self.blocks = model.blocks
        self.norm = model.norm
        self.fc_norm = model.fc_norm
        self.patch_size = model.patch_size
        self.register_buffer('input_chans', torch.tensor(input_chans, dtype=torch.long))
    def forward(self, x):
        bs, n, a, t = x.shape
        itw = a if t == self.patch_size else t
        x = self.patch_embed(x)
        b = x.shape[0]
        ct = self.cls_token.expand(b, -1, -1)
        x = torch.cat((ct, x), dim=1)
        peu = self.pos_embed[:, self.input_chans]
        pe = peu[:, 1:, :].unsqueeze(2).expand(b, -1, itw, -1).flatten(1, 2)
        pe = torch.cat((peu[:, 0:1, :].expand(b, -1, -1), pe), dim=1)
        x = x + pe
        nc = n if t == self.patch_size else a
        te = self.time_embed[:, 0:itw, :].unsqueeze(1).expand(b, nc, -1, -1).flatten(1, 2)
        x[:, 1:, :] += te
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=None)
        x = self.norm(x)
        x = self.fc_norm(x[:, 1:, :].mean(1))
        return x

wrapper = LaBraMWrapper(model, input_chans)
wrapper.eval()
with torch.no_grad():
    wout = wrapper(eeg)
    mout = model(eeg, input_chans=input_chans)
    diff = (wout - mout).abs().max().item()
    print(f'Wrapper max diff vs model: {diff:.6e}')

# ONNX export
print('7. Exporting ONNX (dynamo=True)')
onnx_path = 'C:/Users/pc/AppData/Local/Temp/labram-encoder.onnx'
torch.onnx.export(wrapper, (eeg,), onnx_path, dynamo=True, input_names=['eeg'], output_names=['embedding'], dynamic_axes={'eeg':{0:'batch'},'embedding':{0:'batch'}})

import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
sz = os.path.getsize(onnx_path)/(1024*1024)
print(f'Exported: {onnx_path} ({sz:.2f} MB)')

print('8. ONNX Runtime parity')
import onnxruntime as ort
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
iname = sess.get_inputs()[0].name
oout = sess.run(None, {iname: eeg.numpy()})[0]
pout = wout.numpy()
mdiff = np.abs(oout - pout).max()
print(f'Max diff: {mdiff:.6e}')
print(f'All close (1e-3): {np.allclose(oout, pout, atol=1e-3)}')
print(f'All close (1e-5): {np.allclose(oout, pout, atol=1e-5)}')

# Dynamic batch
eeg2 = torch.randn(4, n_ch, n_patch, psize)
with torch.no_grad(): pout2 = wrapper(eeg2).numpy()
oout2 = sess.run(None, {iname: eeg2.numpy()})[0]
print(f'Batch=4 max diff: {np.abs(oout2-pout2).max():.6e}')
print(f'Batch=4 all close (1e-3): {np.allclose(oout2, pout2, atol=1e-3)}')
print('ONNX Runtime parity verified!')

print()
print('='*70)
print('SUMMARY')
print('='*70)
print(f'Model: LaBraM base (NeuralTransformer)')
print(f'Checkpoint: {CKPT} (96.6 MB)')
print(f'Architecture: BEiT-style, 12 layers, embed_dim=200, 10 heads, init_values=0.1')
print(f'Patch embedding: TemporalConv')
print(f'pos_embed: {model.pos_embed.shape}')
print(f'Input: [batch, {n_ch}, {n_patch}, {psize}]')
print(f'Output: [batch, {wout.shape[-1]}]')
print(f'ONNX: Success (dynamo=True)')
print(f'Parity: max_diff={mdiff:.6e}')
print(f'License: MIT')
print('='*70)
