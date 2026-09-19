"""Laptop ekranina siyah perde (TV'de mac varken, ekranda 1,5 sn onden giden maci gormemek icin).

Perde "katmanli" (layered, %99 opak) pencere: Chrome bunu ortulme saymaz, cizmeye devam eder ve
yakalama surer (tam opak normal pencere Chrome'u durdurur, 2026-09-15 olculdu). Birincil ekrani
kaplar, en ustte durur. Kapatma: Esc ya da cift tik.

    .venv/Scripts/pythonw.exe tools/kara_perde.py
"""
from __future__ import annotations

import ctypes
import tkinter as tk


def main() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:  # noqa: BLE001
        pass
    u = ctypes.windll.user32
    w, h = u.GetSystemMetrics(0), u.GetSystemMetrics(1)  # birincil ekran (laptop paneli)
    r = tk.Tk()
    r.overrideredirect(True)
    r.geometry(f"{w}x{h}+0+0")
    r.configure(bg="black", cursor="none")
    r.attributes("-topmost", True)
    r.attributes("-alpha", 0.99)  # katmanli pencere: Chrome ortulme saymaz
    r.bind("<Escape>", lambda e: r.destroy())
    r.bind("<Double-Button-1>", lambda e: r.destroy())
    r.mainloop()


if __name__ == "__main__":
    main()
