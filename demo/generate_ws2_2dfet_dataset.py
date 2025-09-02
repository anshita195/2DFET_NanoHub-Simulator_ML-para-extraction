import os
import json
from pathlib import Path
import numpy as np
from importlib.machinery import SourceFileLoader
import types


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# (In demo/generate_ws2_2dfet_dataset.py)
# REPLACE the old write_variables_csv function

def write_variables_csv(sample_dir: Path, params: dict, keys_to_write: list) -> None:
    lines = []
    for key in keys_to_write:
        if key in params:
            lines.append(f"{key}={params[key]}\n")
    (sample_dir / "variables.csv").write_text("".join(lines), encoding="utf-8")


def save_idvg(sample_dir: Path, vd_value: float, vgs: np.ndarray, id_a_per_um: np.ndarray) -> None:
    # Must match data/README.md: header "x","y" and values are Vgs [V], Id [uA/um]
    out_path = sample_dir / f"IdVg_Vds={vd_value}.csv"
    # Compose CSV with header an    d rows
    with out_path.open("w", encoding="utf-8") as f:
        f.write('"x","y"\n')
        for x, y in zip(vgs, id_a_per_um):
            f.write(f"{x},{y}\n")


def _load_twodfet(repo_root: Path):
    """Load TwoDFET and its helpers from 2dfets-r10/rappture/main_2DFET.py under Python 3.

    This removes Rappture dependency and makes minimal Python 3 print fixes.
    """
    src_path = repo_root / "2dfets-r10" / "rappture" / "main_2DFET.py"
    text = src_path.read_text(encoding="utf-8")

    # Keep only the pure simulator parts; drop Rappture and UI helpers
    lines = text.splitlines()
    # Remove Rappture import
    lines = [("Rappture = None" if l.strip().startswith("import Rappture") else l) for l in lines]
    # Truncate at get_label (we don't need any UI code; also avoids Python2 prints)
    cut_idx = None
    for i, l in enumerate(lines):
        if l.strip().startswith("def get_label("):
            cut_idx = i
            break
    if cut_idx is not None:
        lines = lines[:cut_idx]
    fixed = "\n".join(lines)

    module = types.ModuleType("twodfet_mod")
    exec(fixed, module.__dict__)
    return module


def run_one_device(sample_dir: Path, params: dict) -> None:
    # Unpack params
    Material = params["Material"]
    e_or_h = params["CMOS"]
    channel_direction = params["channel"]
    tins_nm = params["tins_nm"]
    epsr = params["epsr"]
    Vfb = params["Ef_V"]
    alphag = params["alphag"]
    alphad = params["alphad"]
    transport_model = "with scattering"
    Lg_nm = params["Lg_nm"]
    mfp_nm = params["scatteringmfp_nm"]
    T = params["temperature_K"]

    gVI = params["gVI"]
    gVF = params["gVF"]
    gNV = int(params["gNV"])

    # Build simulator (load module from file path to handle hyphenated folder name)
    repo_root = Path(__file__).resolve().parents[1]
    mod = _load_twodfet(repo_root)
    TwoDFET = getattr(mod, "TwoDFET")

    fet = TwoDFET(
        Material,
        e_or_h,
        channel_direction,
        tins_nm,
        epsr,
        Vfb,
        alphag,
        alphad,
        transport_model,
        Lg_nm,
        mfp_nm,
        T,
    )

    # Gate sweep vector used for saving
    Vgs = np.linspace(gVI, gVF, gNV)

    # Two drain biases as single-point sweeps: 0.1 V and 1.0 V
    for vd in [0.1, 1.0]:
        Vgv, Vdv, Id, *_ = fet.simulate(gVI, gVF, gNV, vd, vd, 1)
        # Id has shape (gNV, 1). The simulator equations yield A/m; units label is uA/um,
        # which is numerically identical to A/m. So no scaling is applied here.
        Id_uA_per_um = Id[:, 0]
        save_idvg(sample_dir, vd, Vgs, Id_uA_per_um)

    # Save variables.csv last
    # write_variables_csv(sample_dir, params)
    # (In demo/generate_ws2_2dfet_dataset.py, inside run_one_device)
# ... (at the end of the function)

# Save variables.csv last, but only the 3 we varied
    varied_params = ["Ef_V", "Lg_nm", "scatteringmfp_nm"]
    write_variables_csv(sample_dir, params, varied_params)


def sample_params(rng: np.random.Generator) -> dict:
    # Confirmed constants
    Material = "WS2"
    CMOS = "e"
    channel = "x"
    tins_nm = 300.0
    epsr = 3.9
    alphag = 0.9
    alphad = 0.03
    temperature_K = 300.0
    gVI = 1.1
    gVF = 50.0
    gNV = 32

    # Varying params
    Ef_V = rng.uniform(0.1, 1.5)
    scatteringmfp_nm = rng.uniform(10.0, 250.0)
    Lg_choices = np.array([300, 500, 750, 1000, 3000, 5000, 10000], dtype=float)
    Lg_nm = float(rng.choice(Lg_choices))

    return {
        "Material": Material,
        "CMOS": CMOS,
        "channel": channel,
        "tins_nm": tins_nm,
        "epsr": epsr,
        "Ef_V": Ef_V,
        "alphag": alphag,
        "alphad": alphad,
        "Lg_nm": Lg_nm,
        "scatteringmfp_nm": scatteringmfp_nm,
        "temperature_K": temperature_K,
        "gVI": gVI,
        "gVF": gVF,
        "gNV": gNV,
        "Vd_values": [0.1, 1.0],
    }


def main():
    # Reproducibility
    rng = np.random.default_rng(19700101)

    repo_root = Path(__file__).resolve().parents[1]
    out_root = repo_root / "data" / "raw"
    ensure_dir(out_root)

    # Number of devices to generate can be set via env var, default 1000
    num_devices = int(os.environ.get("WS2_NUM_DEVICES", "1000"))

    for idx in range(1, num_devices + 1):
        sample_id = f"{idx:05d}"
        sample_dir = out_root / sample_id
        ensure_dir(sample_dir)

        params = sample_params(rng)
        run_one_device(sample_dir, params)

    # Write a small manifest for provenance
    manifest = {
        "num_devices": num_devices,
        "seed": 19700101,
        "material": "WS2",
        "transport_model": "with scattering",
        "output_root": str(out_root),
    }
    (out_root / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


