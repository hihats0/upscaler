"""data/clips/*.info.json -> data/clips/manifest.json (lisans ve atif kaydi).

Sadece Creative Commons Attribution lisansli klipler kabul edilir; digerleri uyariyla listelenmez.
Kullanim: .venv/Scripts/python.exe tools/clip_manifest.py
"""
from __future__ import annotations

import datetime
import glob
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "data", "clips")


def probe(path: str) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name,width,height,r_frame_rate:format=duration,size", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main() -> None:
    path = os.path.join(CLIPS, "manifest.json")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = {c["id"]: c for c in json.load(f)["klipler"]}
    items = []
    for info_path in sorted(glob.glob(os.path.join(CLIPS, "*.info.json"))):
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        vid = info["id"]
        files = [p for p in glob.glob(os.path.join(CLIPS, vid + ".*")) if not p.endswith(".json") and not p.endswith(".part")]
        if not files:
            continue
        lic = info.get("license") or ""
        if "Creative Commons Attribution" not in lic:
            print(f"UYARI {vid}: lisans CC BY degil ({lic!r}), manifeste girmedi")
            continue
        pr = probe(files[0])
        st = pr["streams"][0]
        num, den = map(int, st["r_frame_rate"].split("/"))
        prev = old.get(vid, {})
        items.append({
            "id": vid, "dosya": os.path.basename(files[0]), "url": info.get("webpage_url"),
            "baslik": info.get("title"), "yukleyen": info.get("channel") or info.get("uploader"),
            "kanal_url": info.get("channel_url"), "yukleme_tarihi": info.get("upload_date"),
            "lisans": lic, "lisans_kaynagi": "YouTube izleme sayfasi (yt-dlp license alani; tarayicida sayfa metni de kontrol edildi)",
            "cozunurluk": [st["width"], st["height"]], "fps": round(num / den, 3), "codec": st["codec_name"],
            "sure_sn": round(float(pr["format"]["duration"]), 1), "boyut_mb": round(int(pr["format"]["size"]) / 1e6),
            "alinma_tarihi": prev.get("alinma_tarihi") or datetime.date.today().isoformat(),
            "kullanim": prev.get("kullanim", "olcum+egitim"),
            "atif": f"\"{info.get('title')}\", {info.get('channel')}, {info.get('webpage_url')}, CC BY",
        })
    total = sum(c["boyut_mb"] for c in items)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"not": "Sadece CC BY lisansli, yukleyenin kendi cekimi olan klipler. TOD kaydi yok.",
                   "toplam_mb": total, "klipler": items}, f, ensure_ascii=False, indent=1)
    print(f"{len(items)} klip, {total} MB -> {path}")


if __name__ == "__main__":
    main()
