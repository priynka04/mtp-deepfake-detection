#!/usr/bin/env python3
import sys, re, torch

def find_state_dict(ck):
    if isinstance(ck, dict):
        for k in ("model", "state_dict", "model_state_dict", "net"):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
        if all(hasattr(v, "numel") for v in ck.values() if v is not None):
            return ck
    return None

def report(path, out):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out.append(f"\n===== {path}")

    if isinstance(ck, dict):
        out.append("top-level keys: " + ", ".join(map(str, ck.keys())))
        for k in ("total_steps", "epoch"):
            if k in ck:
                out.append(f"{k}: {ck[k]}")

    sd = find_state_dict(ck)
    if sd is None:
        out.append("could not find a state dict")
        return

    lora = head = other = 0
    n_lora = n_head = n_other = 0

    for k, v in sd.items():
        if not hasattr(v, "numel"):
            continue

        low = k.lower()

        if "lora" in low:
            lora += v.numel()
            n_lora += 1

        elif (re.search(r"(^|\.)fc\.(weight|bias)$", low)
              or "classifier" in low
              or "head" in low) and "mlp" not in low:
            head += v.numel()
            n_head += 1
            out.append(f"  head tensor: {k} {tuple(v.shape)}")

        else:
            other += v.numel()
            n_other += 1

    out.append(f"tensors: {len(sd)} (lora {n_lora}, head {n_head}, other {n_other})")
    out.append(f"LoRA numbers          : {lora:,}")
    out.append(f"head numbers          : {head:,}")
    out.append(f"other (backbone) nums : {other:,}")

    tot = lora + head + other
    out.append(f"total numbers in file : {tot:,}")

    for key in ("optimizer", "optim", "optimizer_state_dict"):
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            od = ck[key]
            out.append(f"optimizer state found under '{key}'")

            groups = od.get("param_groups", [])
            for gi, g in enumerate(groups):
                out.append(
                    f"  group {gi}: {len(g.get('params', []))} parameters, lr={g.get('lr')}"
                )

            st = od.get("state", {})
            sizes = [
                v["exp_avg"].numel()
                for v in st.values()
                if isinstance(v, dict) and "exp_avg" in v
            ]

            out.append(
                f"  parameters that have Adam state: {len(sizes)}, "
                f"total numbers {sum(sizes):,}"
            )
            break
    else:
        out.append("no optimizer state in this file")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_params.py checkpoint.pth")
        sys.exit(1)

    out = []

    for p in sys.argv[1:]:
        report(p, out)

    txt = "\n".join(out)
    print(txt)

    with open("params_report.txt", "w") as f:
        f.write(txt + "\n")
