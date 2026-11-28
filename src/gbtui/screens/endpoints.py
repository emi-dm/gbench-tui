"""Endpoints screen: manage /v1 servers, binary GGUF reader, folder import, local discovery & LoRA picker."""
from __future__ import annotations

import curses
import os
import re
from pathlib import Path

from .. import config, gguf_meta, harness_hub, servers, sysmem
from .. import theme as T
from ..ui import ESC, SIGINT
from .common import ENTER, Field, _adapter_color, _clamp_top, _friendly_probe_err, _quant_color


class EndpointsMixin:
    def screen_endpoints(self):
        ui, sel, top = self.ui, 0, 0
        self.st.probe_all()
        while True:
            eps = self.s.endpoints
            h, w = ui.size()
            self.scr.erase()
            up_n = sum(1 for e in eps if (self.st.ep_ok.get(e.name) or {}).get("ok"))
            ui.header("Endpoints", "OpenAI-compatible /v1 servers, local GGUFs & LoRA adapters",
                      right_badge=(f"● {up_n}/{len(eps)} online", T.GREEN if up_n else T.MUTED))
            sel = max(0, min(sel, len(eps) - 1)) if eps else 0
            detail_h = 10 if eps else 0
            table_y = 3
            table_h = max(4, min(len(eps) + 3, h - detail_h - 6)) if eps else 5
            view = max(1, table_h - 3)
            top = _clamp_top(top, sel, view, len(eps))
            card_r = w - 3

            ui.card(table_y, 2, table_h, w - 4, title="Configured Endpoints", color=T.RULE, title_color=T.CYAN,
                    badge=f"★ {self.s.active}" if self.s.active else "none", badge_color=T.YELLOW)
            hdr_y = table_y + 1
            name_w = 18 if w < 90 else 22
            quant_w = 9 if w >= 88 else 0
            adapt_w = 14 if w >= 106 else 0
            model_w = 15 if w < 100 else 16
            tok_w = 0 if w < 126 else 20
            col_name = 8
            col_quant = col_name + name_w
            col_adapt = col_quant + quant_w
            col_model = col_adapt + adapt_w
            col_tok = col_model + model_w
            col_url = col_tok + tok_w
            hdr_str = (f"{'name':<{name_w}}"
                       + (f"{'quant':<{quant_w}}" if quant_w else "")
                       + (f"{'adapter':<{adapt_w}}" if adapt_w else "")
                       + f"{'model':<{model_w}}"
                       + (f"{'tokenizer':<{tok_w}}" if tok_w else "")
                       + "url")
            ui.put(hdr_y, col_name, hdr_str, ui.color(T.DIM), maxw=max(0, card_r - col_name - 1))
            if not eps:
                ui.put(table_y + 2, 5, "No endpoints yet. Press a add, g GGUF, F import folder, D discover local, or h harness-hub.",
                       ui.color(T.MUTED), maxw=max(0, card_r - 6))
            for i in range(view):
                j = top + i
                if j >= len(eps):
                    break
                e = eps[j]
                ry = table_y + 2 + i
                active = e.name == self.s.active
                if j == sel:
                    ui.sel_bar(ry, 3, w - 6, T.CYAN)
                bg = T.SEL if j == sel else None
                pr = self.st.ep_ok.get(e.name)
                up = bool(pr and pr["ok"])
                ui.put(ry, 4, "●" if up else "○", ui.color(T.GREEN if up else T.DIM, bg, up))
                ui.put(ry, 6, "★" if active else " ", ui.color(T.YELLOW, bg, True))
                ui.put(ry, col_name, f"{e.name[:name_w - 1]:<{name_w}}",
                       ui.color(T.LIGHT if (active or j == sel) else T.SOFT, bg, active or j == sel),
                       maxw=max(0, card_r - col_name - 1))
                if quant_w:
                    eq = e.effective_quant
                    ui.put(ry, col_quant, f"{(eq or '—')[:quant_w - 1]:<{quant_w}}",
                           ui.color(_quant_color(eq), bg, bool(eq)), maxw=max(0, card_r - col_quant - 1))
                if adapt_w:
                    ea = e.effective_adapter or "base"
                    ui.put(ry, col_adapt, f"{ea[:adapt_w - 1]:<{adapt_w}}",
                           ui.color(_adapter_color(ea), bg, ea != "base"), maxw=max(0, card_r - col_adapt - 1))
                ui.put(ry, col_model, f"{(e.model or 'auto')[:model_w - 1]:<{model_w}}",
                       ui.color(T.PALE, bg), maxw=max(0, card_r - col_model - 1))
                if tok_w:
                    ui.put(ry, col_tok, f"{(e.tokenizer or '—')[:tok_w - 1]:<{tok_w}}",
                           ui.color(T.MUTED, bg), maxw=max(0, card_r - col_tok - 1))
                if e.gguf:
                    pid = servers.server_pid(e)
                    tag = ("LoRA ●" if pid else "LoRA ○") if e.lora else ("GGUF ●" if pid else "GGUF ○")
                    tag_col = T.GREEN if pid else (T.PINK if e.lora else T.YELLOW)
                    tag_bold = bool(pid)
                else:
                    tag, tag_col, tag_bold = "HTTP", (T.CYAN if up else T.VIOLET), False
                tag_x = max(col_url + 2, w - len(tag) - 5)
                url_max = max(0, tag_x - col_url - 2)
                if url_max > 0:
                    ui.put(ry, col_url, e.base_url, ui.color(T.CYAN, bg), maxw=url_max)
                if tag_x + len(tag) <= card_r:
                    ui.soft_pill(ry, tag_x - 1, tag, fg=tag_col, bg=T.BAR_ALT, bold=tag_bold)
            ui.scrollbar(table_y + 2, w - 3, view, top, view, len(eps))

            if eps:
                e = eps[sel]
                dy = table_y + table_h + 1
                dh = min(detail_h, h - dy - 2)
                if dh >= 4:
                    pr = self.st.ep_ok.get(e.name)
                    up = bool(pr and pr["ok"])
                    btxt = f"● online · {pr['latency_ms']}ms" if up else "○ offline" if pr else "probing…"
                    ui.card(dy, 2, dh, w - 4, title=e.name, color=T.VIOLET if e.name == self.s.active else T.RULE,
                            title_color=T.PINK if e.name == self.s.active else T.CYAN,
                            badge=btxt, badge_color=T.GREEN if up else T.YELLOW)
                    right_panel = e.gguf and w >= 98 and dh >= 7
                    info_maxw = (w - 48) if right_panel else max(0, w - 19)
                    eq = e.effective_quant
                    q_bits = config.quant_bits(eq)
                    q_desc = (f"{eq}" + (f" (~{q_bits:.1f} bpw)" if q_bits else "") + ("" if e.quant else " · auto")) if eq else "(unspecified)"
                    ea = e.effective_adapter
                    eh = e.effective_harness
                    lora_desc = (f"{Path(e.lora).name}" + (f" (scale {e.lora_scale:g})" if abs(e.lora_scale - 1.0) > 1e-6 else "")
                                 + (f" · {ea or 'lora'}" if ea else "") + (f" → {eh}" if eh else "")) if e.lora else "none (press l to pick a LoRA)"
                    info = [
                        ("url", e.base_url, T.CYAN),
                        ("model", e.model or "(auto: the only model the endpoint serves)", T.SOFT),
                        ("quant", q_desc, _quant_color(eq) if eq else T.DIM),
                        ("lora", lora_desc, _adapter_color(ea) if e.lora else T.DIM),
                        ("tokenizer", e.tokenizer or "(none — needed for Golden chat-template checks and perf)",
                         T.PALE if e.tokenizer else T.MUTED),
                        ("gguf", f"{Path(e.gguf).name}  · ctx {e.ctx} · {e.parallel} slots"
                         if e.gguf else "external server (vLLM / Ollama / LM Studio / API)", T.SOFT if e.gguf else T.DIM),
                    ]
                    if e.ctx_warning:
                        info.append(("warning", "⚠ " + e.ctx_warning, T.YELLOW))
                    if pr:
                        if up:
                            served = ", ".join(pr["models"][:5]) or "(empty)"
                            mismatch = e.model and pr["models"] and e.model not in pr["models"]
                            p_txt = f"✔ serves {served}" + (f"  ⚠ '{e.model}' not listed" if mismatch else "")
                            info.append(("probe", p_txt, T.YELLOW if mismatch else T.GREEN))
                        else:
                            info.append(("probe", _friendly_probe_err(e, pr["error"]), T.YELLOW if e.gguf else T.RED))
                    for i, (k_, v, col_) in enumerate(info):
                        if dy + 1 + i >= dy + dh - 1:
                            break
                        ui.put(dy + 1 + i, 4, f"{k_:<10}", ui.color(T.DIM))
                        ui.put(dy + 1 + i, 15, v, ui.color(col_), maxw=max(0, info_maxw))
                    if right_panel:
                        rx = w - 31
                        for row_i in range(1, dh - 1):
                            ui.put(dy + row_i, rx - 2, "│", ui.color(T.RULE_DIM))
                        est = servers.estimate_bytes(e)
                        tot = self.st.mem_total or 1
                        ui.put(dy + 1, rx, "RAM Footprint", ui.color(T.PALE, None, True))
                        ui.put(dy + 2, rx, f"~{sysmem.fmt(est)} estimated", ui.color(T.CYAN, None, True))
                        ui.heat_bar(dy + 3, rx, 22, min(1.0, est / tot))
                        ui.put(dy + 4, rx, f"host free: {sysmem.fmt(self.st.mem_avail)}", ui.color(T.MUTED))
                        pid = servers.server_pid(e)
                        ui.put(dy + 5, rx, f"● pid {pid} running" if pid else "○ press s to launch",
                               ui.color(T.GREEN if pid else T.YELLOW))

            ui.footer("Enter edit · Space active · l pick LoRA · h hub · g GGUF · F folder · D discover · s start · t test · d del · q back")
            self.scr.refresh()
            k = ui.key(500)
            self.tick += 1
            if k is None or k == "RESIZE":
                continue
            if k in ("q", ESC, SIGINT):
                return
            if k in (curses.KEY_UP, "k") and eps:
                sel = (sel - 1) % len(eps)
            elif k in (curses.KEY_DOWN, "j") and eps:
                sel = (sel + 1) % len(eps)
            elif k == curses.KEY_PPAGE and eps:
                sel = max(0, sel - view)
            elif k == curses.KEY_NPAGE and eps:
                sel = min(len(eps) - 1, sel + view)
            elif k == "a":
                before = len(self.s.endpoints)
                self.edit_endpoint(None)
                if len(self.s.endpoints) > before:
                    sel = len(self.s.endpoints) - 1
            elif k == "g":
                before = len(self.s.endpoints)
                self.add_gguf()
                if len(self.s.endpoints) > before:
                    sel = len(self.s.endpoints) - 1
            elif k == "F":
                before = len(self.s.endpoints)
                self.import_gguf_folder()
                if len(self.s.endpoints) > before:
                    sel = len(self.s.endpoints) - 1
            elif k == "D":
                before = len(self.s.endpoints)
                self.discover_local_servers()
                if len(self.s.endpoints) > before:
                    sel = len(self.s.endpoints) - 1
            elif k == "h":
                before = len(self.s.endpoints)
                self.browse_harness_hub(eps[sel] if eps else None)
                if len(self.s.endpoints) > before:
                    sel = len(self.s.endpoints) - 1
            elif k == "r":
                self.st.probe_all()
                ui.notify("probing all endpoints…", T.CYAN, 1.8)
            elif not eps:
                continue
            elif k == "l":
                self.select_lora_for_endpoint(eps[sel])
            elif k in ENTER:
                self.edit_endpoint(eps[sel])
            elif k == " ":
                self.s.active = eps[sel].name
                self.save()
                self.st.kick()
                ui.notify(f"★ active endpoint set to {eps[sel].name}", T.YELLOW)
            elif k == "d":
                if ui.confirm(f"Delete endpoint {eps[sel].name}? (its saved results stay)"):
                    servers.stop_server(eps[sel])
                    name = eps[sel].name
                    del eps[sel]
                    self.st.ep_ok.pop(name, None)
                    sel = max(0, sel - 1)
                    if self.s.active not in [e.name for e in eps]:
                        self.s.active = eps[0].name if eps else ""
                    self.save()
                    ui.notify(f"deleted endpoint {name}", T.YELLOW)
            elif k == "s":
                e = eps[sel]
                if not e.gguf:
                    ui.message("This endpoint has no GGUF: start its server yourself (vLLM, Ollama, LM Studio…).")
                elif servers.server_pid(e):
                    servers.stop_server(e)
                    ui.notify(f"■ llama-server for {e.name} stopped", T.YELLOW)
                else:
                    why = ui.busy(f"starting llama-server for {e.name}…", servers.start_server, e, self.s.min_free_gb)
                    if why:
                        ui.message(why, title="llama-server", color=T.RED)
                    else:
                        ui.notify(f"✔ llama-server ready on {e.base_url}", T.GREEN, 3.0)
                self.st.probe_all()
            elif k == "t":
                self.test_endpoint(eps[sel])

    def _local_lora_candidates(self) -> list[Path]:
        """Find cached harness-hub LoRAs + any *lora*.gguf in common folders."""
        seen: dict[str, Path] = {}
        for p in harness_hub.local_loras():
            seen[str(p.resolve())] = p
        for d in (Path.home(), Path.home() / "Downloads", Path.home() / "models"):
            if d.is_dir():
                for p in d.glob("*lora*.gguf"):
                    seen[str(p.resolve())] = p
                for p in d.glob("*adapter*.gguf"):
                    seen[str(p.resolve())] = p
        return sorted(seen.values(), key=lambda p: p.name.lower())

    def pick_lora_path(self, current: str = "") -> str | None:
        """Interactive LoRA selector usable both as a form picker and standalone from Endpoints."""
        ui = self.ui
        cands = self._local_lora_candidates()
        items = [("", "(none — base model weights)", "clear LoRA adapter")]
        for p in cands:
            ad = config.detect_adapter(p.name) or "lora"
            hr = config.detect_harness(p.name)
            sz = f"{p.stat().st_size / 1e6:.0f} MB" if p.exists() else ""
            note = " · ".join(x for x in (ad, f"→ {hr}" if hr else "", sz, str(p.parent)) if x)
            items.append((str(p), p.name, note))
        items.append(("__custom__", "Enter custom .gguf path…", "browse / type path manually"))
        sel_idx = next((i for i, it in enumerate(items) if it[0] == current), 0)
        choice = ui.choose("Select LoRA GGUF adapter", items, sel=sel_idx)
        if choice is None:
            return None
        if choice == "__custom__":
            v = ui.edit("LoRA GGUF path", current or str(harness_hub.lora_cache_dir()))
            return v.strip() if v is not None else None
        return choice

    def select_lora_for_endpoint(self, ep: config.Endpoint):
        """Attach, swap, or clear a LoRA on `ep` (or clone into a new LoRA endpoint) directly in the TUI."""
        ui = self.ui
        if not ep.gguf:
            ui.message("LoRA GGUF adapters require a local GGUF endpoint (llama-server).", title="LoRA Selector", color=T.YELLOW)
            return
        cands = self._local_lora_candidates()
        items = [
            ("__clear__", "(none — base weights)", "detach LoRA from this endpoint"),
            ("__hub__", "↓ Download from huggingface.co/harness-hub…", f"filter for {ep.base_model}"),
        ]
        for p in cands:
            ad = config.detect_adapter(p.name) or "lora"
            hr = config.detect_harness(p.name)
            sz = f"{p.stat().st_size / 1e6:.0f} MB" if p.exists() else ""
            items.append((str(p), p.name, " · ".join(x for x in (ad, f"→ {hr}" if hr else "", sz) if x)))
        items.append(("__custom__", "Custom .gguf path…", "type path manually"))
        choice = ui.choose(f"LoRA for {ep.name}", items)
        if choice is None:
            return
        if choice == "__hub__":
            self.browse_harness_hub(ep)
            return
        if choice == "__clear__":
            if servers.server_pid(ep):
                servers.stop_server(ep)
            ep.lora = ""
            ep.adapter_name = ""
            ep.target_harness = ""
            self.save()
            self.st.probe_all()
            ui.notify(f"✔ cleared LoRA on {ep.name} (base weights)", T.GREEN)
            return
        if choice == "__custom__":
            choice = ui.edit("LoRA GGUF path", ep.lora or str(harness_hub.lora_cache_dir()))
            if not choice:
                return
        ad = config.detect_adapter(choice)
        hr = config.detect_harness(choice)
        mode = ui.choose(f"Attach {Path(choice).name}", [
            ("clone", f"Create / update variant '{ep.name}-{ad or 'lora'}'", "keeps base endpoint intact (recommended for comparisons)"),
            ("inplace", f"Attach directly to '{ep.name}'", "modifies this endpoint in place"),
        ])
        if mode is None:
            return
        if mode == "clone":
            base_slug = ep.name
            new_name = f"{base_slug}-{ad}" if ad else f"{base_slug}-lora"
            existing = self.s.endpoint(new_name)
            target = existing or config.Endpoint(
                name=new_name, base_url=ep.base_url, model=new_name, tokenizer=ep.tokenizer,
                quant=ep.effective_quant, gguf=ep.gguf, ctx=max(ep.ctx, 65536), parallel=ep.parallel,
            )
            target.lora = choice
            target.adapter_name = ad
            target.target_harness = hr
            target.ctx = max(target.ctx, 65536)
            if not existing:
                self.s.endpoints.append(target)
            self.s.active = target.name
            self.save()
            self.st.probe_all()
            ui.notify(f"✔ active LoRA endpoint: {target.name} ({ad or 'lora'}, ctx {target.ctx})", T.GREEN, 3.0)
        else:
            if servers.server_pid(ep):
                servers.stop_server(ep)
            ep.lora = choice
            ep.adapter_name = ad
            ep.target_harness = hr
            if ep.ctx < 32768:
                ep.ctx = 65536
            self.save()
            self.st.probe_all()
            ui.notify(f"✔ attached {Path(choice).name} to {ep.name} (ctx {ep.ctx})", T.GREEN, 3.0)

    def discover_local_servers(self):
        """Auto-discover running Ollama, LM Studio, llama-server, and vLLM endpoints on common localhost ports."""
        ui = self.ui
        found = ui.busy("probing localhost (Ollama :11434, LM Studio :1234, llama-server :8080, vLLM :8000)…",
                        servers.discover_local)
        if not found:
            ui.message("No running OpenAI-compatible servers found on common localhost ports\n"
                       "(checked :11434, :1234, :8080, :8081, :8000, :5000).",
                       title="Auto-discover local servers", color=T.YELLOW)
            return
        items = []
        for idx, f in enumerate(found):
            models_str = ", ".join(f["models"][:3]) or "(no models listed)"
            items.append((idx, f"{f['provider']} · {f['base_url']}", f"{f['latency_ms']}ms · {models_str}"))
        picked = self.multi_pick("Discovered local servers (Space toggles · Enter adds)", items, set(range(len(found))))
        if not picked:
            return
        added = 0
        for idx in picked:
            f = found[idx]
            model_id = f["models"][0] if f["models"] else ""
            slug_src = model_id or f["provider"].lower().replace(" ", "-")
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug_src.lower()).strip(".-") or "local-server"
            if any(e.name == slug for e in self.s.endpoints):
                slug = f"{slug}-{f['port']}"
            ep = config.Endpoint(
                name=slug,
                base_url=f["base_url"],
                model=model_id,
                quant=config.detect_quant(model_id),
                notes=f"auto-discovered {f['provider']}",
            )
            self.s.endpoints.append(ep)
            self.s.active = self.s.active or ep.name
            added += 1
        self.save()
        self.st.probe_all()
        ui.notify(f"✔ added {added} discovered server(s)", T.GREEN, 3.0)

    def import_gguf_folder(self):
        """Scan a folder of .gguf files using the binary GGUF header reader (arch, params, ctx, exact quant)."""
        ui = self.ui
        default_dir = str(Path.home() / "models") if (Path.home() / "models").is_dir() else str(Path.home())
        folder_s = ui.edit("Folder containing .gguf models", default_dir,
                           hint="reads binary GGUF headers (arch, params, native ctx, exact quant)")
        if not folder_s:
            return
        folder = Path(folder_s).expanduser()
        if not folder.is_dir():
            ui.message(f"Not a directory: {folder}", title="Import GGUF folder", color=T.RED)
            return
        base_models, lora_adapters = ui.busy(f"reading GGUF headers in {folder.name}…", gguf_meta.scan_folder, folder, True)
        if not base_models:
            ui.message(f"No base .gguf models found in {folder}\n(found {len(lora_adapters)} LoRA adapter(s)).",
                       title="Import GGUF folder", color=T.YELLOW)
            return
        items = [(i, g.path.name, g.summary) for i, g in enumerate(base_models)]
        picked = self.multi_pick(f"Import GGUF models from {folder.name} (Space toggles)", items, set(range(len(base_models))))
        if not picked:
            return
        added = 0
        for i in picked:
            g = base_models[i]
            base, fn_quant = config.split_model_quant(g.path.stem)
            quant = g.quant or fn_quant
            base_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", (g.name or base).lower()).strip(".-") or "gguf-model"
            name = f"{base_slug}-{quant.lower()}" if quant and quant.lower() not in base_slug else base_slug
            if any(e.name == name for e in self.s.endpoints):
                continue
            port = 8080
            ctx = min(65536, g.context_length) if g.context_length else 16384
            ep = config.Endpoint(
                name=name,
                base_url=f"http://127.0.0.1:{port}/v1",
                model=name,
                quant=quant,
                gguf=str(g.path),
                ctx=max(8192, ctx),
                notes=g.summary,
            )
            self.s.endpoints.append(ep)
            self.s.active = self.s.active or ep.name
            added += 1
        self.save()
        self.st.probe_all()
        ui.notify(f"✔ imported {added} GGUF model(s) from {folder.name}", T.GREEN, 3.0)

    def test_endpoint(self, e):
        ui = self.ui
        pr = ui.busy(f"GET {e.base_url}/models…", servers.probe, e)
        if not pr["ok"]:
            ui.message(f"✗ {e.base_url} not reachable: {pr['error']}" + ("\n\nStart its llama-server with s." if e.gguf else ""),
                       title="Endpoint test", color=T.RED)
            return
        chat_model = e.model or (pr["models"][0] if pr["models"] else None)
        r = ui.busy("sending one chat completion…", servers.chat_once, e, model=chat_model)
        txt = (f"✔ serves: {', '.join(pr['models'])} ({pr['latency_ms']} ms)\n\n"
               + (f"✔ chat: “{r['text'][:200]}” ({r['seconds']}s)" if r["ok"] else f"✗ chat: {r['text']}"))
        if e.model and pr["models"] and e.model not in pr["models"]:
            txt += f"\n\n⚠ '{e.model}' is not in the served ids: gbench matches by name, or set the model to one of them."
        ui.message(txt, title=f"Endpoint test · {e.name}", color=T.GREEN if r["ok"] else T.YELLOW)

    def add_gguf(self):
        ui = self.ui
        found = sorted({str(p) for d in (Path.home(), Path.home() / "Downloads", Path.home() / "models")
                        if d.is_dir() for p in d.glob("*.gguf") if "lora" not in p.name.lower()})
        items = []
        for p in found:
            info = gguf_meta.read_gguf(p)
            if info:
                items.append((p, Path(p).name, info.summary))
            else:
                q = config.detect_quant(p)
                sz = f"{os.path.getsize(p) / 1e9:.1f} GB"
                items.append((p, Path(p).name, f"{q} · {sz}" if q else sz))
        items.append(("", "other path…", ""))
        path = ui.choose("Select GGUF file", items)
        if path is None:
            return
        if path == "":
            path = ui.edit("path to the .gguf", str(Path.home()))
            if not path:
                return
        info = gguf_meta.read_gguf(path)
        base, fn_quant = config.split_model_quant(Path(path).stem)
        quant = (info.quant if info and info.quant else fn_quant)
        base_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", base.lower()).strip(".") or "gguf-model"
        name = f"{base_slug}-{quant.lower()}" if quant else base_slug
        port = 8080 + len([e for e in self.s.endpoints if e.gguf])
        tok = ui.edit("tokenizer (HF id, for gbench's --tokenizer)", f"LiquidAI/{base}" if base.upper().startswith("LFM") else "",
                      hint="the original model's repo, e.g. LiquidAI/LFM3-1.7B; empty = none")
        ctx = min(65536, info.context_length) if (info and info.context_length) else 16384
        ep = config.Endpoint(name=name, base_url=f"http://127.0.0.1:{port}/v1", model=name, tokenizer=tok or "",
                             quant=quant, gguf=path, ctx=max(8192, ctx), notes=(info.summary if info else ""))
        if any(e.name == ep.name for e in self.s.endpoints):
            ep.name += f"-{port}"
        self.s.endpoints.append(ep)
        self.s.active = self.s.active or ep.name
        self.save()
        self.st.probe_all()
        q_msg = f" ({quant})" if quant else ""
        ui.notify(f"✔ added GGUF endpoint {ep.name}{q_msg}", T.GREEN)

    def browse_harness_hub(self, base_ep=None):
        """Browse & download GGUF LoRAs from Hugging Face `harness-hub` and attach to a base GGUF endpoint."""
        ui = self.ui
        base_filter = base_ep.base_model if (base_ep and base_ep.gguf) else ""
        try:
            adapters = ui.busy("querying huggingface.co/harness-hub…", harness_hub.fetch_gguf_adapters)
        except Exception as e:                       # noqa: BLE001
            ui.message(f"Could not query Hugging Face harness-hub:\n\n{e}", title="harness-hub", color=T.RED)
            return
        if not adapters:
            ui.message("No GGUF LoRA adapters found in harness-hub.", title="harness-hub", color=T.YELLOW)
            return
        compat = harness_hub.compatible_adapters(base_filter, adapters) if base_filter else adapters
        pool = compat or adapters
        items = []
        if base_filter and compat:
            items.append(("__all__", f"Show all {len(adapters)} harness-hub adapters…", f"currently filtered to {base_filter}"))
        for a in pool:
            tag = "✔ cached" if a["cached_path"] else "↓ download"
            items.append((a, f"{a['filename']}", f"{a['harness'] or 'lora'} · {a['base_model']} · {tag}"))
        choice = ui.choose(f"harness-hub LoRAs ({base_filter or 'all'})", items)
        if choice is None:
            return
        if choice == "__all__":
            items = [(a, a["filename"], f"{a['harness'] or 'lora'} · {a['base_model']} · {'✔ cached' if a['cached_path'] else '↓'}")
                     for a in adapters]
            choice = ui.choose("All harness-hub GGUF LoRAs", items)
            if choice is None:
                return
        dest_box = {"path": None}

        def _dl(say):
            dest_box["path"] = harness_hub.download_lora(choice["repo_id"], choice["filename"], on_line=say)
            return True

        self.live_task(f"Downloading {choice['filename']}", _dl)
        dest = dest_box["path"]
        if not dest:
            return
        if base_ep and base_ep.gguf:
            ad = choice["adapter_name"] or config.detect_adapter(str(dest), target_harness=choice["harness"])
            new_name = f"{base_ep.name}-{ad}" if ad else f"{base_ep.name}-lora"
            existing = self.s.endpoint(new_name)
            ep = existing or config.Endpoint(
                name=new_name, base_url=base_ep.base_url, model=new_name, tokenizer=base_ep.tokenizer,
                quant=base_ep.effective_quant, gguf=base_ep.gguf, ctx=max(base_ep.ctx, 65536), parallel=base_ep.parallel,
            )
            ep.lora = str(dest)
            ep.adapter_name = ad
            ep.target_harness = choice["harness"]
            ep.ctx = max(ep.ctx, 65536)
            if not existing:
                self.s.endpoints.append(ep)
            self.save()
            self.st.probe_all()
            ui.notify(f"✔ ready: {ep.name} ({ad}, ctx {ep.ctx})", T.GREEN, 3.0)
        else:
            ui.notify(f"✔ downloaded to {dest}", T.GREEN, 3.0)

    def edit_endpoint(self, e):
        new = e is None
        e = e or config.Endpoint(name=f"endpoint-{len(self.s.endpoints) + 1}")
        fields = [
            Field("name", "Name", help="unique identifier (no / or ..)"),
            Field("base_url", "Base URL (/v1)", help="what gbench/Harbor gets as --remote-endpoint / --base-url"),
            Field("model", "Model id (--models)", help="the id the endpoint serves; empty = its only model"),
            Field("quant", "Quantization", help="e.g. Q4_K_M, Q8_0, BF16, FP8, AWQ (auto-detected if empty)"),
            Field("tokenizer", "Tokenizer (--tokenizer)", help="HF id of the original model, e.g. LiquidAI/LFM3-1.7B"),
            Field("api_key_env", "API key env var", help="name of the environment variable with the key (never the key)"),
            Field("gguf", "GGUF file (optional)", help="gbench-tui starts llama-server for it on the URL's port"),
            Field("lora", "LoRA GGUF (picker)", "picker", [self.pick_lora_path],
                  help="Enter to pick from cached/local *-lora*.gguf adapters or clear to base"),
            Field("lora_scale", "LoRA scale", "float", help="1.0 = --lora; other values pass --lora-scaled", min_val=0.0),
            Field("adapter_name", "Adapter label", help="e.g. pi-lora, opencode-lora, hermes-lora (auto if empty)"),
            Field("target_harness", "Target harness", help="pi, opencode, hermes, super-adapter (auto if empty)"),
            Field("ctx", "llama-server ctx", "int", help="total context (32768–65536+ recommended for agent harnesses)", min_val=512),
            Field("parallel", "llama-server slots", "int", help="concurrent requests (each gets ctx/slots)", min_val=1),
            Field("extra_args", "llama-server extra args"),
            Field("notes", "Notes"),
        ]
        vals = {f.key: getattr(e, f.key) for f in fields}
        old = e.name
        while self.form(("New endpoint" if new else f"Endpoint · {old}"), fields, vals, done_label="Save"):
            new_name = str(vals["name"] or "").strip()
            if not new_name or "/" in new_name or "\\" in new_name or new_name in (".", ".."):
                self.ui.message("Endpoint name must be non-empty and cannot contain '/' or '..'.", color=T.RED)
                continue
            others = {x.name for x in self.s.endpoints if x is not e}
            if new_name in others:
                self.ui.message(f"An endpoint named '{new_name}' already exists.", color=T.RED)
                continue
            vals["name"] = new_name
            vals["quant"] = config.normalize_quant(vals.get("quant", ""))
            vals["adapter_name"] = config.normalize_adapter(vals.get("adapter_name", ""))
            vals["target_harness"] = config.normalize_harness(vals.get("target_harness", ""))
            if not new and new_name != old and servers.server_pid(e):
                servers.stop_server(e)
            for k, v in vals.items():
                setattr(e, k, v)
            if new:
                self.s.endpoints.append(e)
            elif new_name != old:
                self.st.ep_ok.pop(old, None)
            if self.s.active in ("", old):
                self.s.active = e.name
            self.save()
            self.st.probe_all()
            if e.ctx_warning:
                self.ui.notify(f"✔ saved {e.name} · ⚠ {e.ctx_warning}", T.YELLOW, 4.0)
            else:
                self.ui.notify(f"✔ saved endpoint {e.name}", T.GREEN)
            return
