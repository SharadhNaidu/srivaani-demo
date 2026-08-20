from __future__ import annotations

import sys
import tkinter as tk


def main():
    root = tk.Tk()
    root.title("PASTE TARGET")
    root.geometry("560x220+120+120")
    text = tk.Text(root, font=("Nirmala UI", 14), wrap="word")
    text.pack(fill="both", expand=True)
    text.insert("1.0", "PREFIX>")
    text.focus_force()
    text.mark_set("insert", "end")

    def dump():
        content = text.get("1.0", "end-1c")
        sys.stdout.write("CONTENT:" + content.replace("\n", "\\n") + "\n")
        sys.stdout.flush()
        root.after(400, dump)

    root.after(400, dump)
    root.after(60000, root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
