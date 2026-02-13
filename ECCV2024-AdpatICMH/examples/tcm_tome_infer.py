import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in (_PROJECT_DIR, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.dataloader import Kodak  # noqa: E402
from examples.models.tic import TIC  # noqa: E402
from examples.models.TIC_TOME import TIC_TOME  # noqa: E402
from examples.models.tinylic import TinyLIC  # noqa: E402
from examples.models.TinyLIC_TOME import TinyLIC_TOME  # noqa: E402
from examples.models.tcm import TCM  # noqa: E402
from examples.models.TCM_TOME import TCM_TOME  # noqa: E402


MODEL_REGISTRY = {
    "TIC": TIC,
    "TIC_TOME": TIC_TOME,
    "TinyLIC": TinyLIC,
    "TinyLIC_TOME": TinyLIC_TOME,
    "TCM": TCM,
    "TCM_TOME": TCM_TOME,
}

_MODEL_DEFAULT_NM = {
    "TIC": (128, 192),
    "TinyLIC": (128, 320),
    "TCM": (128, 320),
}

_TCM_DEFAULTS = {
    "tcm_config": [2, 2, 2, 2, 2, 2],
    "head_dim": [8, 16, 32, 32, 16, 8],
    "drop_path_rate": 0.0,
    "Z": 192,
    "num_slices": 5,
    "max_support_slices": 5,
}


def _normalize_model_family(name: str) -> str:
    k = str(name).strip().lower()
    aliases = {
        "tcm": "TCM",
        "tcm_tome": "TCM",
        "tic": "TIC",
        "tic_tome": "TIC",
        "tinylic": "TinyLIC",
        "tinylic_tome": "TinyLIC",
        "tinytic": "TinyLIC",
        "tinytic_tome": "TinyLIC",
    }
    if k not in aliases:
        raise ValueError("Unknown model family. Use one of: TCM, TIC, TinyLIC")
    return aliases[k]


def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def parse_args(argv=None):
    parser = argparse.ArgumentParser("tcm_tome_infer")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config.")
    cli = parser.parse_args(argv)

    with open(cli.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("YAML top-level must be a dict.")

    args = SimpleNamespace(**cfg)
    args.config = cli.config

    if not hasattr(args, "name") or not args.name:
        args.name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not hasattr(args, "out_dir"):
        args.out_dir = "./inference_results"
    if not hasattr(args, "dataset_kodak"):
        raise ValueError("Missing required key: dataset_kodak")
    if not hasattr(args, "model"):
        args.model = "TCM"
    if not hasattr(args, "checkpoint"):
        args.checkpoint = ""
    if not hasattr(args, "checkpoint_base"):
        args.checkpoint_base = args.checkpoint
    if not hasattr(args, "checkpoint_tic"):
        args.checkpoint_tic = args.checkpoint
    if not hasattr(args, "checkpoint_tinylic"):
        args.checkpoint_tinylic = args.checkpoint
    if not hasattr(args, "checkpoint_tcm"):
        args.checkpoint_tcm = args.checkpoint
    if not hasattr(args, "checkpoint_tic_tome"):
        args.checkpoint_tic_tome = args.checkpoint
    if not hasattr(args, "checkpoint_tinylic_tome"):
        args.checkpoint_tinylic_tome = args.checkpoint
    if not hasattr(args, "checkpoint_tcm_tome"):
        args.checkpoint_tcm_tome = args.checkpoint
    if not hasattr(args, "checkpoint_tome"):
        args.checkpoint_tome = args.checkpoint

    family = _normalize_model_family(args.model)
    if not hasattr(args, "N"):
        args.N = _MODEL_DEFAULT_NM[family][0]
    if not hasattr(args, "M"):
        args.M = _MODEL_DEFAULT_NM[family][1]
    for k, v in _TCM_DEFAULTS.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    if not hasattr(args, "enc_tome"):
        args.enc_tome = True
    if not hasattr(args, "dec_tome"):
        args.dec_tome = True
    if not hasattr(args, "h_tome"):
        args.h_tome = False
    if not hasattr(args, "tome_config"):
        args.tome_config = None

    if not hasattr(args, "cuda"):
        args.cuda = True
    if not hasattr(args, "gpu_id"):
        args.gpu_id = 0
    if not hasattr(args, "num_workers"):
        args.num_workers = 4
    if not hasattr(args, "test_batch_size"):
        args.test_batch_size = 1
    if not hasattr(args, "seed"):
        args.seed = 42

    if not hasattr(args, "eval_forward"):
        args.eval_forward = True
    if not hasattr(args, "eval_codec"):
        args.eval_codec = True
    if not hasattr(args, "update_entropy"):
        args.update_entropy = True
    if not hasattr(args, "save_recon"):
        args.save_recon = True

    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(args):
    use_cuda = bool(args.cuda) and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{int(args.gpu_id)}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return device


def build_model(args, name):
    name = str(name)
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}', choices: {list(MODEL_REGISTRY.keys())}")

    cls = MODEL_REGISTRY[name]
    if name.startswith("TCM"):
        tcm_kwargs = {
            "config": args.tcm_config,
            "head_dim": args.head_dim,
            "drop_path_rate": args.drop_path_rate,
            "Z": args.Z,
            "num_slices": args.num_slices,
            "max_support_slices": args.max_support_slices,
        }
        if name == "TCM_TOME":
            return cls(N=args.N, M=args.M, args=args, **tcm_kwargs)
        return cls(N=args.N, M=args.M, **tcm_kwargs)
    if name in {"TIC_TOME", "TinyLIC_TOME"}:
        return cls(N=args.N, M=args.M, args=args)
    return cls(N=args.N, M=args.M)


def _strip_prefix(state_dict, prefix):
    out = {}
    for k, v in state_dict.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out


def load_checkpoint(model, ckpt_path, device):
    if not ckpt_path:
        logging.warning("No checkpoint provided, using random-initialized weights.")
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif "net" in ckpt and isinstance(ckpt["net"], dict):
            state_dict = ckpt["net"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = _strip_prefix(state_dict, "module.")
    load_ret = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = [], []
    if load_ret is None:
        # Some local model classes override load_state_dict and do not return IncompatibleKeys.
        pass
    elif hasattr(load_ret, "missing_keys") and hasattr(load_ret, "unexpected_keys"):
        missing = list(load_ret.missing_keys)
        unexpected = list(load_ret.unexpected_keys)
    elif isinstance(load_ret, tuple) and len(load_ret) == 2:
        missing, unexpected = load_ret

    logging.info("Loaded checkpoint: %s", ckpt_path)
    if missing:
        logging.info("Missing keys: %d", len(missing))
    if unexpected:
        logging.info("Unexpected keys: %d", len(unexpected))


def _tic_safe_decompress(self, strings, shape):
    """Runtime fix for TIC/TIC_TOME decompress clamp bug in the upstream model file."""
    assert isinstance(strings, list) and len(strings) == 2
    z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
    gaussian_params = self.h_s(z_hat)
    scales_hat, means_hat = gaussian_params.chunk(2, 1)
    indexes = self.gaussian_conditional.build_indexes(scales_hat)
    y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
    x_hat, _ = self.g_s(y_hat)
    x_hat = x_hat.clamp_(0, 1)
    return {"x_hat": x_hat}


def _patch_runtime_model_quirks(model, family):
    if family == "TIC":
        model.decompress = MethodType(_tic_safe_decompress, model)


def _sync_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _nested_bytes_len(x):
    if isinstance(x, (bytes, bytearray)):
        return len(x)
    if isinstance(x, (list, tuple)):
        return sum(_nested_bytes_len(v) for v in x)
    return 0


def _psnr(x_hat, x):
    mse = F.mse_loss(x_hat, x)
    mse = torch.clamp(mse, min=1e-12)
    return float((10.0 * torch.log10(1.0 / mse)).item())


@torch.no_grad()
def evaluate(args, model, model_tag, loader, device, out_dir):
    model.eval()
    if bool(args.update_entropy):
        try:
            model.update(force=True)
        except Exception as e:
            logging.warning("model.update(force=True) failed: %s", e)

    per_image = []
    sum_forward_bpp = 0.0
    sum_forward_psnr = 0.0
    sum_forward_ms = 0.0
    cnt_forward = 0

    sum_codec_bpp = 0.0
    sum_codec_psnr = 0.0
    sum_enc_ms = 0.0
    sum_dec_ms = 0.0
    cnt_codec = 0

    save_recon = bool(args.save_recon)
    recon_dir = os.path.join(out_dir, "recon", model_tag)
    if save_recon:
        os.makedirs(recon_dir, exist_ok=True)

    for idx, x in enumerate(loader):
        x = x.to(device, non_blocking=True)
        n, _, h, w = x.shape
        num_pixels = n * h * w
        item = {"index": int(idx), "h": int(h), "w": int(w)}

        if bool(args.eval_forward):
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            out = model(x)
            _sync_if_cuda(device)
            t1 = time.perf_counter()

            x_hat = out["x_hat"].clamp(0, 1)
            bpp = 0.0
            for likelihood in out["likelihoods"].values():
                bpp += float((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)).item())
            psnr = _psnr(x_hat, x)
            ms = (t1 - t0) * 1000.0

            sum_forward_bpp += bpp
            sum_forward_psnr += psnr
            sum_forward_ms += ms
            cnt_forward += 1

            item["forward_bpp"] = bpp
            item["forward_psnr"] = psnr
            item["forward_ms"] = ms

            if save_recon:
                save_image(x_hat, os.path.join(recon_dir, f"{idx:04d}_forward.png"))

        if bool(args.eval_codec):
            _sync_if_cuda(device)
            t0 = time.perf_counter()
            enc = model.compress(x)
            _sync_if_cuda(device)
            t1 = time.perf_counter()

            dec = model.decompress(enc["strings"], enc["shape"])
            _sync_if_cuda(device)
            t2 = time.perf_counter()

            x_hat = dec["x_hat"].clamp(0, 1)
            bits = _nested_bytes_len(enc["strings"]) * 8
            bpp = bits / float(num_pixels)
            psnr = _psnr(x_hat, x)
            enc_ms = (t1 - t0) * 1000.0
            dec_ms = (t2 - t1) * 1000.0

            sum_codec_bpp += bpp
            sum_codec_psnr += psnr
            sum_enc_ms += enc_ms
            sum_dec_ms += dec_ms
            cnt_codec += 1

            item["codec_bpp"] = bpp
            item["codec_psnr"] = psnr
            item["enc_ms"] = enc_ms
            item["dec_ms"] = dec_ms

            if save_recon:
                save_image(x_hat, os.path.join(recon_dir, f"{idx:04d}_codec.png"))

        per_image.append(item)
        logging.info("[%s] image=%03d %s", model_tag, idx, item)

    summary = {}
    if cnt_forward > 0:
        summary["forward_avg_bpp"] = sum_forward_bpp / cnt_forward
        summary["forward_avg_psnr"] = sum_forward_psnr / cnt_forward
        summary["forward_avg_ms"] = sum_forward_ms / cnt_forward
    if cnt_codec > 0:
        summary["codec_avg_bpp"] = sum_codec_bpp / cnt_codec
        summary["codec_avg_psnr"] = sum_codec_psnr / cnt_codec
        summary["codec_avg_enc_ms"] = sum_enc_ms / cnt_codec
        summary["codec_avg_dec_ms"] = sum_dec_ms / cnt_codec
        summary["codec_avg_total_ms"] = (sum_enc_ms + sum_dec_ms) / cnt_codec

    return {"summary": summary, "per_image": per_image}


def _pick_checkpoint(args, family, use_tome):
    fam = family.lower()
    if use_tome:
        candidates = [
            getattr(args, f"checkpoint_{fam}_tome", ""),
            getattr(args, "checkpoint_tome", ""),
            getattr(args, f"checkpoint_{fam}", ""),
        ]
    else:
        candidates = [
            getattr(args, f"checkpoint_{fam}", ""),
            getattr(args, "checkpoint_base", ""),
        ]
    candidates.append(getattr(args, "checkpoint", ""))
    for p in candidates:
        if isinstance(p, str) and p.strip():
            return p
    return ""


def compare_time(results_base, results_tome):
    s_tcm = results_base.get("summary", {})
    s_tome = results_tome.get("summary", {})
    out = {}

    if "forward_avg_ms" in s_tcm and "forward_avg_ms" in s_tome:
        tcm_ms = float(s_tcm["forward_avg_ms"])
        tome_ms = float(s_tome["forward_avg_ms"])
        out["forward_base_ms"] = tcm_ms
        out["forward_tome_ms"] = tome_ms
        out["forward_speedup_base_over_tome"] = (tcm_ms / max(tome_ms, 1e-12))

    if "codec_avg_total_ms" in s_tcm and "codec_avg_total_ms" in s_tome:
        tcm_ms = float(s_tcm["codec_avg_total_ms"])
        tome_ms = float(s_tome["codec_avg_total_ms"])
        out["codec_total_base_ms"] = tcm_ms
        out["codec_total_tome_ms"] = tome_ms
        out["codec_total_speedup_base_over_tome"] = (tcm_ms / max(tome_ms, 1e-12))

    return out


def main(argv=None):
    args = parse_args(argv)
    set_seed(int(args.seed))
    family = _normalize_model_family(args.model)
    base_name = family
    tome_name = f"{family}_TOME"

    out_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "infer.log")
    setup_logger(log_path)

    device = setup_device(args)
    logging.info("Config: %s", args.config)
    logging.info("Output dir: %s", out_dir)
    logging.info("Device: %s", device)

    data_tf = transforms.Compose([transforms.ToTensor()])
    dataset = Kodak(args.dataset_kodak, data_tf)
    loader = DataLoader(
        dataset,
        batch_size=int(args.test_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    logging.info("Kodak images: %d", len(dataset))

    if not bool(args.save_recon):
        logging.info("save_recon=False in config, force set to True for dual-model x_hat export.")
        args.save_recon = True

    logging.info("Model family: %s", family)
    logging.info("=== Evaluate baseline model: %s ===", base_name)
    model_tcm = build_model(args, base_name).to(device)
    _patch_runtime_model_quirks(model_tcm, family)
    load_checkpoint(model_tcm, _pick_checkpoint(args, family, use_tome=False), device)
    results_tcm = evaluate(args, model_tcm, base_name, loader, device, out_dir)

    del model_tcm
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logging.info("=== Evaluate accelerated model: %s ===", tome_name)
    model_tome = build_model(args, tome_name).to(device)
    _patch_runtime_model_quirks(model_tome, family)
    load_checkpoint(model_tome, _pick_checkpoint(args, family, use_tome=True), device)
    results_tome = evaluate(args, model_tome, tome_name, loader, device, out_dir)

    compare = compare_time(results_tcm, results_tome)
    results = {
        base_name: results_tcm,
        tome_name: results_tome,
        "time_comparison": compare,
    }
    result_path = os.path.join(out_dir, "metrics.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info("%s summary: %s", base_name, results_tcm["summary"])
    logging.info("%s summary: %s", tome_name, results_tome["summary"])
    logging.info("Time comparison: %s", compare)
    logging.info("Saved metrics: %s", result_path)


if __name__ == "__main__":
    main()
