"""Capture trusted local neural outputs on one checkout, verify on another.

Only synthetic forward/backward calls, never fitting or official benchmark data.
Serialized modules are unsafe for untrusted inputs: verification requires an
explicit --trust-local-checkpoints flag. Keep the generated tensors out of git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def _fusion_case(method, entry, torch):
    torch.manual_seed(147)
    kwargs = {entry.inputs_param.value: entry.resolve_input_count(2), "d_model": 8, "dropout": 0.0}
    if entry.is_extended:
        kwargs.update(num_latents=2, num_heads=2, residual_hidden_dim=8)
    return entry.fusion_class(**kwargs).eval(), kwargs


def capture(output, torch):
    from fedot_ind.core.models.future.mapping import FUSION_REGISTRY
    from fedot_ind.core.multimodal.enums import MultimodalModality as Modality
    from fedot_ind.core.models.nn.network_impl.encoders.builder import build_encoder
    from fedot_ind.core.models.nn.network_impl.encoders.transformation_presets import (
        raw_encoder_config, stats_encoder_config, gaf_encoder_config, stft_encoder_config,
    )
    from fedot_ind.core.models.nn.network_modules.layers.attention_layers import MultiHeadAttention

    records = {}
    for method, entry in FUSION_REGISTRY.items():
        model, kwargs = _fusion_case(method, entry, torch)
        x, y = (torch.randn(2, 8, requires_grad=True) for _ in range(2))
        result = model.fuse({Modality.raw: x, Modality.stats: y}, (Modality.raw, Modality.stats),
                            raw_modality=Modality.raw)
        result.square().sum().backward()
        records[method.value] = dict(kwargs=kwargs, state=model.state_dict(), x=x.detach(), y=y.detach(),
                                     result=result.detach(), xgrad=x.grad, ygrad=y.grad)
        torch.save(model, output / f"{method.value}_full.pt")
    for name, config, shape in (
        ("raw", raw_encoder_config(2, d_model=8), (2, 2, 16)),
        ("stats", stats_encoder_config(5, d_model=8), (2, 5)),
        ("gaf", gaf_encoder_config(1, d_model=8), (2, 1, 8, 8)),
        ("stft", stft_encoder_config(1, d_model=8), (2, 1, 9, 5)),
    ):
        torch.manual_seed(349)
        model = build_encoder(config).eval()
        x = torch.randn(*shape, requires_grad=True)
        result = model(x)
        result.square().sum().backward()
        records["encoder_" + name] = dict(config=config, state=model.state_dict(), x=x.detach(),
                                          result=result.detach(), xgrad=x.grad)
        torch.save(model, output / f"encoder_{name}_full.pt")
    torch.manual_seed(511)
    model = MultiHeadAttention(8, 2, res_attention=True).eval()
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    result = model(x, attn_mask=mask)
    result[0].square().sum().backward()
    records["attention"] = dict(state=model.state_dict(), x=x.detach(), mask=mask,
                                result=tuple(t.detach() for t in result), xgrad=x.grad)
    torch.save(records, output / "records.pt")
    return {"phase": "capture", "cases": len(records), "torch": torch.__version__}


def verify(output, torch):
    from fedot_ind.core.models.future.mapping import FUSION_REGISTRY, FusionMethod
    from fedot_ind.core.multimodal.enums import MultimodalModality as Modality
    from fedot_ind.neural.encoders import build_encoder
    from fedot_ind.neural.blocks.attention import MultiHeadAttention

    records = torch.load(output / "records.pt", weights_only=False, map_location="cpu")
    results = []
    for name, record in records.items():
        x = record["x"].clone().requires_grad_()
        if name == "attention":
            torch.manual_seed(511)
            model = MultiHeadAttention(8, 2, res_attention=True).eval()
        elif name.startswith("encoder_"):
            torch.manual_seed(349)
            model = build_encoder(record["config"]).eval()
        else:
            torch.manual_seed(147)
            entry = FUSION_REGISTRY[FusionMethod(name)]
            model = entry.fusion_class(**record["kwargs"]).eval()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, record["state"][key], rtol=0, atol=0)
        model.load_state_dict(record["state"], strict=True)
        if name == "attention":
            values = model(x, attn_mask=record["mask"])
            for actual, expected in zip(values, record["result"]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual, expected = values[0], record["result"][0]
        elif name.startswith("encoder_"):
            actual, expected = model(x), record["result"]
        else:
            y = record["y"].clone().requires_grad_()
            actual = model.fuse({Modality.raw: x, Modality.stats: y}, (Modality.raw, Modality.stats),
                                raw_modality=Modality.raw)
            expected = record["result"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        torch.testing.assert_close(x.grad, record["xgrad"], rtol=0, atol=0)
        if "ygrad" in record:
            torch.testing.assert_close(y.grad, record["ygrad"], rtol=0, atol=0)
        assert list(model.state_dict()) == list(record["state"])
        full_path = output / f"{name}_full.pt"
        if full_path.exists():
            restored = torch.load(full_path, weights_only=False, map_location="cpu")
            restored_state = restored.state_dict()
            assert list(restored_state) == list(record["state"])
            for key, value in restored_state.items():
                torch.testing.assert_close(value, record["state"][key], rtol=0, atol=0)
            if name.startswith("encoder_"):
                restored_result = restored.eval()(record["x"])
            else:
                restored_result = restored.eval().fuse(
                    {Modality.raw: record["x"], Modality.stats: record["y"]},
                    (Modality.raw, Modality.stats), raw_modality=Modality.raw,
                )
            torch.testing.assert_close(restored_result, expected, rtol=0, atol=0)
        results.append({"name": name, "state_entries": len(record["state"]),
                        "initialization_matches": True,
                        "output_shape": list(actual.shape), "output_max_absolute_error": 0,
                        "input_gradient_max_absolute_error": 0, "full_module_restored": full_path.exists()})
    return {"phase": "verify", "torch": torch.__version__, "cases": results,
            "reference_sha256": hashlib.sha256((output / "records.pt").read_bytes()).hexdigest(),
            "scope": "Synthetic forward/backward and trusted local checkpoint compatibility; no training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "verify"))
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--trust-local-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.phase == "verify" and not args.trust_local_checkpoints:
        parser.error("Verification reads pickle; explicitly trust your own locally generated checkpoints")
    if args.phase == "capture" and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Refusing to overwrite a nonempty capture directory")
    sys.path.insert(0, str(args.source_dir.resolve()))
    import torch
    if args.phase == "capture":
        args.output_dir.mkdir(parents=True, exist_ok=True)
    result = capture(args.output_dir, torch) if args.phase == "capture" else verify(args.output_dir, torch)
    text = json.dumps(result, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
