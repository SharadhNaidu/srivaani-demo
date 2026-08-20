from __future__ import annotations

import threading
import time

try:
    import win32api
    import win32clipboard
    import win32con
    import win32gui
    import win32process
    HAVE_WIN32 = True
except Exception:
    HAVE_WIN32 = False


class FocusTarget:
    def __init__(self, hwnd=None, title="", pid=None):
        self.hwnd = hwnd
        self.title = title
        self.pid = pid

    @property
    def valid(self):
        if not HAVE_WIN32 or not self.hwnd:
            return False
        try:
            return bool(win32gui.IsWindow(self.hwnd))
        except Exception:
            return False

    def __bool__(self):
        return self.valid


def capture_focus():
    if not HAVE_WIN32:
        return FocusTarget()
    try:
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return FocusTarget()
        title = win32gui.GetWindowText(hwnd) or ""
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return FocusTarget(hwnd, title, pid)
    except Exception:
        return FocusTarget()


def process_name(pid):
    if not HAVE_WIN32 or not pid:
        return ""
    try:
        import win32process as wp
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        try:
            path = wp.GetModuleFileNameEx(handle, 0)
        finally:
            win32api.CloseHandle(handle)
        return path.rsplit("\\", 1)[-1]
    except Exception:
        return ""


def _restore_foreground(hwnd):
    if not HAVE_WIN32 or not hwnd:
        return False
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        if win32gui.GetForegroundWindow() == hwnd:
            return True

        target_thread, _ = win32process.GetWindowThreadProcessId(hwnd)
        current_thread = win32api.GetCurrentThreadId()
        attached = False
        if target_thread and target_thread != current_thread:
            try:
                attached = bool(win32process.AttachThreadInput(current_thread, target_thread, True))
            except Exception:
                attached = False
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            try:
                win32gui.BringWindowToTop(hwnd)
            except Exception:
                pass
        finally:
            if attached:
                try:
                    win32process.AttachThreadInput(current_thread, target_thread, False)
                except Exception:
                    pass
        for _ in range(20):
            if win32gui.GetForegroundWindow() == hwnd:
                return True
            time.sleep(0.01)
        return win32gui.GetForegroundWindow() == hwnd
    except Exception:
        return False


def _clipboard_open(retries=12, delay=0.02):
    for _ in range(retries):
        try:
            win32clipboard.OpenClipboard()
            return True
        except Exception:
            time.sleep(delay)
    return False


def get_clipboard():
    if not HAVE_WIN32:
        try:
            import pyperclip
            return pyperclip.paste()
        except Exception:
            return None
    if not _clipboard_open():
        return None
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
            return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        return None
    except Exception:
        return None
    finally:
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass


def set_clipboard(text):
    if not HAVE_WIN32:
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except Exception:
            return False
    if not _clipboard_open():
        return False
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
        return True
    except Exception:
        return False
    finally:
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass


def _send_paste():
    try:
        from pynput.keyboard import Controller, Key
        kb = Controller()
        with kb.pressed(Key.ctrl):
            kb.press("v")
            kb.release("v")
        return True
    except Exception:
        pass
    if not HAVE_WIN32:
        return False
    try:
        VK_CONTROL, VK_V = 0x11, 0x56
        win32api.keybd_event(VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(VK_V, 0, 0, 0)
        win32api.keybd_event(VK_V, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        return True
    except Exception:
        return False


def type_text(text, delay=0.004):
    try:
        from pynput.keyboard import Controller
        kb = Controller()
        for ch in text:
            if ch == "\n":
                kb.type("\r")
            else:
                kb.type(ch)
            time.sleep(delay)
        return True
    except Exception:
        return False


class Injector:
    def __init__(self, settings, own_hwnds=None):
        self.settings = settings
        self.own_hwnds = own_hwnds or (lambda: set())
        self._lock = threading.Lock()

    def is_own_window(self, target):
        try:
            return bool(target and target.hwnd in self.own_hwnds())
        except Exception:
            return False

    def deliver(self, text, target=None):
        result = {"copied": False, "pasted": False, "restored": False,
                  "target": getattr(target, "title", "") or "", "reason": ""}
        if not text:
            result["reason"] = "empty"
            return result

        if self.settings.get("auto_copy", True):
            result["copied"] = set_clipboard(text)

        if not self.settings.get("auto_paste", True):
            result["reason"] = "paste_disabled"
            return result

        if target is None or not target.valid:
            result["reason"] = "no_target"
            return result

        if self.is_own_window(target):
            result["reason"] = "own_window"
            return result

        with self._lock:
            previous = get_clipboard() if not result["copied"] else None
            if not result["copied"] and not set_clipboard(text):
                result["reason"] = "clipboard_failed"
                return result

            result["restored"] = _restore_foreground(target.hwnd)
            if not result["restored"]:
                result["reason"] = "focus_failed"
                return result

            time.sleep(0.04)
            result["pasted"] = _send_paste()
            if not result["pasted"]:
                result["pasted"] = type_text(text)
                result["reason"] = "typed_fallback"

            if previous is not None:
                def restore_clip():
                    time.sleep(0.6)
                    set_clipboard(previous)
                threading.Thread(target=restore_clip, daemon=True).start()

        return result
