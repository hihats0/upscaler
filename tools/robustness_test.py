"""Dayaniklilik sinavi (Hat 1.4 paket 5).

ffplay'de oynayan kendi test klibimiz uzerinde watch calisirken sirayla:
donma (surec askiya), simge durumu, pencere boyutu degisimi (1280x720 ve geri), pencere kapanip
yeniden acilmasi, ekran degisimi benzetimi (sunucu yeniden acilir). Sonunda watch'in cikis kodu,
olay sayilari, 10 sn'lik cikis FPS'leri ve VRAM/RAM basi-sonu raporlanir.

Kullanim:
    .venv/Scripts/python.exe tools/robustness_test.py --run-name robust1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from av_sync_test import TITLE, find_window, launch_player  # noqa: E402


def wait_event(path: str, kind: str, timeout: float = 60.0) -> dict | None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec["olay"] == kind:
                        return rec
        time.sleep(0.3)
    return None


def main() -> None:
    import psutil
    import win32con
    import win32gui
    sys.path.insert(0, ROOT)
    from upscaler import winutil
    winutil.dpi_aware()  # SetWindowPos fiziksel piksel (Windows %125 olcekte 1920 -> 2400 oluyordu)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=time.strftime("robust_%Y%m%d_%H%M%S"))
    ap.add_argument("--clip", default=os.path.join(ROOT, "data", "av_test_1080p50.mp4"))
    args = ap.parse_args()
    run_dir = os.path.join(ROOT, "runs", args.run_name)
    events = os.path.join(run_dir, "events.jsonl")
    player = launch_player(args.clip)
    find_window(TITLE)
    time.sleep(1.0)
    watch = subprocess.Popen([sys.executable, "-m", "upscaler", "watch", "--title", TITLE, "--title-must", "",
                              "--seconds", "185", "--run-name", args.run_name, "--info"], cwd=ROOT)
    log = []

    def step(label: str) -> None:
        rec = (round(time.time() - t0, 1), label)
        log.append(rec)
        print(f"[test {rec[0]:6.1f}s] {label}", flush=True)

    try:
        if wait_event(events, "ilk_kare", 60) is None:
            raise SystemExit("watch ilk kareyi vermedi")
        t0 = time.time()
        step("ilk kare geldi, 15 sn normal")
        time.sleep(15)

        step("DONMA: ffplay sureci 8 sn askida")
        pp = psutil.Process(player.pid)
        pp.suspend()
        time.sleep(8)
        pp.resume()
        step("donma bitti")
        time.sleep(12)

        hwnd = find_window(TITLE)
        step("SIMGE DURUMU: 6 sn")
        win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
        time.sleep(6)
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        step("geri acildi")
        time.sleep(12)

        step("BOYUT: 1280x720")
        win32gui.SetWindowPos(hwnd, 0, 0, 0, 1280, 720, win32con.SWP_NOZORDER)
        time.sleep(12)
        step("BOYUT: 2560x1440 (1080p'den buyuk: tampon 1080p'de kalmali)")
        win32gui.SetWindowPos(hwnd, 0, 0, 0, 2560, 1440, win32con.SWP_NOZORDER)
        time.sleep(12)
        win32gui.SetWindowPos(hwnd, 0, 0, 0, 1920, 1080, win32con.SWP_NOZORDER)
        step("BOYUT: 1920x1080 geri")
        time.sleep(12)

        step("KAPANMA: ffplay oldu, 6 sn sonra yeniden")
        player.kill()
        time.sleep(6)
        player = launch_player(args.clip)
        find_window(TITLE)
        step("ffplay yeniden acildi")
        time.sleep(15)

        step("EKRAN DEGISIMI benzetimi (sunucu yeniden acilir)")
        open(os.path.join(run_dir, "reopen.flag"), "w").close()
        time.sleep(15)
        step("bitis bekleniyor")
        rc = watch.wait(timeout=120)
    finally:
        if watch.poll() is None:
            watch.kill()
        player.kill()

    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    with open(os.path.join(run_dir, "stats.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"\nwatch cikis kodu: {rc}")
    print("test adimlari:", log)
    print("olaylar:", s["olaylar"])
    print("10 sn cikis fps:", [r["cikis_fps"] for r in rows])
    print("kaynak durumu:", [r["kaynak"] for r in rows])
    print("yol:", sorted({r["yol"] for r in rows}))
    print("vram reserved bas/son MB:", rows[0]["vram_reserved_mb"], rows[-1]["vram_reserved_mb"],
          "| smi:", rows[0]["vram_smi_mb"], rows[-1]["vram_smi_mb"], "| rss:", rows[0]["rss_mb"], rows[-1]["rss_mb"])
    vram0, vram1 = float(rows[0]["vram_smi_mb"]), float(rows[-1]["vram_smi_mb"])
    checks = {
        "cikis kodu 0": rc == 0,
        "islem hatasi 0": s.get("islem_hatasi") == 0,
        "hata olayi yok": not any("hata" in k for k in s["olaylar"]),
        "VRAM buyumesi < 100 MB": vram1 - vram0 < 100,
        "son 3 pencere >= 59.5 FPS": all(float(r["cikis_fps"]) >= 59.5 for r in rows[-3:]),
        "yol sonunda motor 1920x1080": rows[-1]["yol"] == "motor 1920x1080",
    }
    for k, v in checks.items():
        print(f"[{'GECTI' if v else 'KALDI'}] {k}")
    for k in ("cikis_fps_aktif", "gec_tik", "bekleme_tik", "tutma_tik", "islem_hatasi", "tampon_sifirlama",
              "yeniden_damgalama", "uctan_uca_ms_p50_p95_p99", "gec_sunum_orani"):
        print(f"{k}: {s.get(k)}")


if __name__ == "__main__":
    main()
