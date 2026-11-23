#!/usr/bin/env python3
"""Open the latest solved image from the log and report sky coordinates on click."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path

from comet_catalogs import compare_archives, dec_dms, describe_match, ra_hms

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - message shown to the user
    raise SystemExit("Install Pillow first: py -m pip install Pillow") from exc


CATALOG_DIR = Path(__file__).resolve().parent / "catalogs"
DEFAULT_MPC_CATALOG = CATALOG_DIR / "mpc_13p_observations.txt"
DEFAULT_HORIZONS_CATALOG = CATALOG_DIR / "horizons_13p_icrf.txt"


def latest_log_record(log_path: Path) -> dict[str, object]:
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read log file {log_path}: {exc}") from exc
    blocks = re.split(r"(?m)^={20,}\s*$", text)
    for block in reversed(blocks):
        image_match = re.search(r"(?m)^Image:\s*(.+?)\s*$", block)
        center_match = re.search(
            r"Center\s*\(RA,\s*Dec\):\s*\(?\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)",
            block,
            re.IGNORECASE,
        )
        if not image_match or not center_match:
            continue
        size_match = re.search(
            r"Size:\s*([\d.]+)\s*x\s*([\d.]+)\s*deg", block, re.IGNORECASE
        )
        orientation_match = re.search(
            r"Up\s+is\s+([+-]?[\d.]+)\s+degrees?\s+E\s+of\s+N",
            block,
            re.IGNORECASE,
        )
        wcs_match = re.search(r"(?m)^Local WCS file:\s*(.+?)\s*$", block)
        pixel_scale_match = re.search(
            r"Pixel scale:\s*([\d.]+)\s*arcsec/pixel", block, re.IGNORECASE
        )
        return {
            "image": Path(image_match.group(1).strip().strip('"')),
            "ra": float(center_match.group(1)),
            "dec": float(center_match.group(2)),
            "width_deg": float(size_match.group(1)) if size_match else None,
            "height_deg": float(size_match.group(2)) if size_match else None,
            "orientation_deg": float(orientation_match.group(1)) if orientation_match else 0.0,
            "pixel_scale": float(pixel_scale_match.group(1)) if pixel_scale_match else None,
            "wcs_file": Path(wcs_match.group(1).strip().strip('"')) if wcs_match else None,
        }
    raise RuntimeError(
        "No complete calibration record found in the log. Expected Image and Center (RA, Dec) lines."
    )


def read_fits_header(path: Path) -> dict[str, object]:
    header: dict[str, object] = {}
    with path.open("rb") as stream:
        found_end = False
        while not found_end:
            block = stream.read(2880)
            if not block:
                break
            for offset in range(0, len(block), 80):
                card = block[offset : offset + 80].decode("ascii", errors="ignore")
                key = card[:8].strip()
                if key == "END":
                    found_end = True
                    break
                if not key or card[8:10] != "= ":
                    continue
                raw = card[10:].split("/", 1)[0].strip()
                if raw.startswith("'"):
                    value: object = raw.strip("'").strip()
                elif raw in {"T", "F"}:
                    value = raw == "T"
                else:
                    try:
                        value = float(raw.replace("D", "E"))
                    except ValueError:
                        value = raw
                header[key] = value
    if not header:
        raise RuntimeError(f"No FITS header found in {path}")
    return header


def wcs_matrix(header: dict[str, object]) -> tuple[float, float, float, float]:
    keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    if all(key in header for key in keys):
        return tuple(float(header[key]) for key in keys)  # type: ignore[return-value]
    if "CDELT1" not in header or "CDELT2" not in header:
        raise RuntimeError("WCS FITS header does not contain a CD matrix or CDELT values")
    cdelt1, cdelt2 = float(header["CDELT1"]), float(header["CDELT2"])
    return (
        cdelt1 * float(header.get("PC1_1", 1.0)),
        cdelt1 * float(header.get("PC1_2", 0.0)),
        cdelt2 * float(header.get("PC2_1", 0.0)),
        cdelt2 * float(header.get("PC2_2", 1.0)),
    )


def tangent_to_sky(ra0_deg: float, dec0_deg: float, xi_deg: float, eta_deg: float) -> tuple[float, float]:
    """Convert tangent-plane offsets (gnomonic/TAN) to RA and Dec."""
    ra0, dec0 = math.radians(ra0_deg), math.radians(dec0_deg)
    xi, eta = math.radians(xi_deg), math.radians(eta_deg)
    rho = math.hypot(xi, eta)
    if rho == 0:
        return ra0_deg % 360.0, dec0_deg
    c = math.atan(rho)
    sin_c, cos_c = math.sin(c), math.cos(c)
    dec = math.asin(cos_c * math.sin(dec0) + eta * sin_c * math.cos(dec0) / rho)
    ra = ra0 + math.atan2(
        xi * sin_c,
        rho * math.cos(dec0) * cos_c - eta * math.sin(dec0) * sin_c,
    )
    return math.degrees(ra) % 360.0, math.degrees(dec)


class CoordinateViewer:
    def __init__(self, log_path: Path, record: dict[str, object]):
    def __init__(
        self,
        log_path: Path,
        record: dict[str, object],
        mpc_catalog: Path,
        horizons_catalog: Path,
    ):
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.log_path = log_path
        self.record = record
        self.mpc_catalog = mpc_catalog
        self.horizons_catalog = horizons_catalog
        self.selected_position: dict[str, float] | None = None
        self.image_path = Path(record["image"])
        self.image = Image.open(self.image_path).convert("RGB")
        self.width, self.height = self.image.size
        self.wcs: dict[str, object] | None = None
        self.mapping_description = "Approximate mapping from center/size/orientation"
        wcs_path = record.get("wcs_file")
        if isinstance(wcs_path, Path) and wcs_path.is_file():
            try:
                header = read_fits_header(wcs_path)
                if "TAN" not in str(header.get("CTYPE1", "")).upper():
                    raise RuntimeError("The WCS projection is not TAN")
                self.wcs = header
                self.mapping_description = "WCS FITS solution"
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"Could not use the WCS file: {exc}")

        self.root = tk.Tk()
        self.root.title(f"Image coordinate viewer — {self.image_path.name}")
        self.scale = min(1.0, 1100 / self.width, 760 / self.height)
        self.photo = None
        self.image_item = None

        toolbar = tk.Frame(self.root, padx=8, pady=6)
        toolbar.pack(fill="x")
        tk.Button(toolbar, text="−", width=3, command=lambda: self.zoom(0.8)).pack(side="left")
        tk.Button(toolbar, text="+", width=3, command=lambda: self.zoom(1.25)).pack(side="left", padx=(4, 10))
        tk.Button(toolbar, text="Fit", command=self.fit).pack(side="left")
        self.info = tk.Label(toolbar, text=self.mapping_description, anchor="w")
        self.info.pack(side="left", padx=12, fill="x", expand=True)

        area = tk.Frame(self.root)
        area.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(area, background="#202020", highlightthickness=0)
        self.scroll_x = tk.Scrollbar(area, orient="horizontal", command=self.canvas.xview)
        self.scroll_y = tk.Scrollbar(area, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scroll_y.grid(row=0, column=1, sticky="ns")
        self.scroll_x.grid(row=1, column=0, sticky="ew")
        area.rowconfigure(0, weight=1)
        area.columnconfigure(0, weight=1)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        self.status = tk.StringVar(value="Click a point in the image to read its coordinates.")
        actions = tk.Frame(self.root, padx=8, pady=4)
        actions.pack(fill="x")
        self.save_button = tk.Button(
            actions,
            text="Save position and compare with both catalogs",
            command=self.save_and_compare,
            state="disabled",
        )
        self.save_button.pack(side="left")
        tk.Label(self.root, textvariable=self.status, anchor="w", padx=8, pady=6).pack(fill="x")
        self.render()

    def render(self) -> None:
        from PIL import ImageTk

        display_width = max(1, round(self.width * self.scale))
        display_height = max(1, round(self.height * self.scale))
        resized = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.configure(
            scrollregion=(0, 0, display_width, display_height),
            width=min(display_width, 1100),
            height=min(display_height, 760),
        )

    def zoom(self, factor: float) -> None:
        self.scale = max(0.03, min(8.0, self.scale * factor))
        self.render()

    def fit(self) -> None:
        self.scale = min(1.0, 1100 / self.width, 760 / self.height)
        self.render()

    def on_wheel(self, event) -> None:
        self.zoom(1.2 if event.delta > 0 else 1 / 1.2)

    def pixel_to_sky(self, x: float, y: float) -> tuple[float, float]:
        if self.wcs:
            header = self.wcs
            cd11, cd12, cd21, cd22 = wcs_matrix(header)
            dx = (x + 1.0) - float(header["CRPIX1"])
            dy = (y + 1.0) - float(header["CRPIX2"])
            xi = cd11 * dx + cd12 * dy
            eta = cd21 * dx + cd22 * dy
            return tangent_to_sky(float(header["CRVAL1"]), float(header["CRVAL2"]), xi, eta)

        width_deg = self.record.get("width_deg")
        height_deg = self.record.get("height_deg")
        if width_deg is None or height_deg is None:
            raise RuntimeError("The log has no Size row, and no local WCS file is available")
        width_tan = 2 * math.tan(math.radians(float(width_deg)) / 2)
        height_tan = 2 * math.tan(math.radians(float(height_deg)) / 2)
        x_right = (x - (self.width - 1) / 2) * width_tan / self.width
        y_up = ((self.height - 1) / 2 - y) * height_tan / self.height
        angle = math.radians(float(self.record.get("orientation_deg") or 0.0))
        east = math.cos(angle) * x_right + math.sin(angle) * y_up
        north = -math.sin(angle) * x_right + math.cos(angle) * y_up
        return tangent_to_sky(
            float(self.record["ra"]), float(self.record["dec"]),
            math.degrees(east), math.degrees(north),
        )

    @staticmethod
    def format_ra(ra_deg: float) -> str:
        seconds = (ra_deg % 360.0) / 15.0 * 3600.0
        hour = int(seconds // 3600)
        minute = int((seconds - hour * 3600) // 60)
        second = seconds - hour * 3600 - minute * 60
        return f"{hour:02d}h {minute:02d}m {second:06.3f}s"

    @staticmethod
    def format_dec(dec_deg: float) -> str:
        sign = "+" if dec_deg >= 0 else "−"
        seconds = abs(dec_deg) * 3600
        degree = int(seconds // 3600)
        minute = int((seconds - degree * 3600) // 60)
        second = seconds - degree * 3600 - minute * 60
        return f"{sign}{degree:02d}° {minute:02d}′ {second:06.3f}″"

    def on_click(self, event) -> None:
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x = canvas_x / self.scale - 0.5
        y = canvas_y / self.scale - 0.5
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        try:
            ra, dec = self.pixel_to_sky(x, y)
        except (KeyError, RuntimeError, ValueError) as exc:
            self.messagebox.showerror("Coordinate conversion", str(exc))
            return
        result = (
            f"RA {self.format_ra(ra)}   Dec {self.format_dec(dec)}   "
            f"({ra:.6f}°, {dec:.6f}°)   Pixel ({x + 1:.1f}, {y + 1:.1f})"
        )
        self.selected_position = {"ra": ra, "dec": dec, "x": x + 1, "y": y + 1}
        self.save_button.configure(state="normal")
        self.status.set(result)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"Click {datetime.now().astimezone().isoformat(timespec='seconds')}: "
                f"{result}\n"

    def save_and_compare(self) -> None:
        if self.selected_position is None:
            return
        position = self.selected_position
        ra = position["ra"]
        dec = position["dec"]
        timestamp = datetime.now().astimezone()
        stamp = timestamp.isoformat(timespec="seconds")
        ra_text = self.format_ra(ra)
        dec_text = self.format_dec(dec)
        csv_path = self.log_path.with_name("measured_positions.csv")
        comparison_fields = [
            "closest_archive",
            "closest_code",
            "closest_time_utc",
            "separation_arcmin",
            "mpc_separation_arcmin",
            "horizons_separation_arcmin",
        ]
        row = {
            "timestamp": stamp,
            "image": str(self.image_path),
            "pixel_x_1_based": f"{position['x']:.2f}",
            "pixel_y_1_based": f"{position['y']:.2f}",
            "ra_deg": f"{ra:.8f}",
            "dec_deg": f"{dec:.8f}",
            "ra_hms": ra_text,
            "dec_dms": dec_text,
            "closest_archive": "",
            "closest_code": "",
            "closest_time_utc": "",
            "separation_arcmin": "",
            "mpc_separation_arcmin": "",
            "horizons_separation_arcmin": "",
        }
        fields = list(row)
        log_lines = [
            f"Measured position saved: {stamp}",
            f"Image: {self.image_path}",
            f"Pixel (FITS 1-based): ({position['x']:.2f}, {position['y']:.2f})",
            f"Coordinates: {ra_text}, {dec_text} ({ra:.8f}°, {dec:.8f}°)",
        ]
        try:
            matches = compare_archives(
                ra, dec, self.mpc_catalog, self.horizons_catalog
            )
            mpc = matches["mpc"]
            horizons = matches["horizons"]
            winner = min(
                (mpc, horizons), key=lambda match: match.separation_arcmin
            )
            winner_name = winner.observation.archive
            report_lines = [
                f"MPC nearest: {mpc.observation.moment:%Y-%m-%d %H:%M:%S} UT — "
                f"RA {ra_hms(mpc.observation.ra_deg)}, Dec {dec_dms(mpc.observation.dec_deg)} — "
                f"Δ {mpc.separation_arcmin:.3f}′ — "
                f"{self._code_for(mpc.observation.moment)}",
                f"JPL Horizons nearest: {horizons.observation.moment:%Y-%m-%d %H:%M:%S} UT — "
                f"RA {ra_hms(horizons.observation.ra_deg)}, "
                f"Dec {dec_dms(horizons.observation.dec_deg)} — "
                f"Δ {horizons.separation_arcmin:.3f}′ — "
                f"{self._code_for(horizons.observation.moment)}",
                f"Closest archive: {winner_name}",
                *describe_match("Selected match", winner),
            ]
            log_lines.extend(["Archive comparison:", *report_lines])
            row.update(
                {
                    "closest_archive": winner_name,
                    "closest_code": self._code_for(winner.observation.moment),
                    "closest_time_utc": winner.observation.moment.isoformat(),
                    "separation_arcmin": f"{winner.separation_arcmin:.6f}",
                    "mpc_separation_arcmin": f"{mpc.separation_arcmin:.6f}",
                    "horizons_separation_arcmin": f"{horizons.separation_arcmin:.6f}",
                }
            )
            self.show_comparison("\n".join(report_lines))
        except (OSError, ValueError) as exc:
            log_lines.append(f"Archive comparison failed: {exc}")
            self.messagebox.showerror("Catalog comparison", str(exc))
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            previous_rows: list[dict[str, str]] = []
            old_fields: list[str] | None = None
            if csv_path.exists() and csv_path.stat().st_size:
                with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                    reader = csv.DictReader(csv_file)
                    old_fields = reader.fieldnames
                    previous_rows = [dict(old) for old in reader]
            if old_fields and old_fields != fields:
                with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fields)
                    writer.writeheader()
                    for old in previous_rows:
                        writer.writerow({key: old.get(key, "") for key in fields})
                    writer.writerow(row)
            else:
                with csv_path.open("a", encoding="utf-8-sig", newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fields)
                    if not old_fields:
                        writer.writeheader()
                    writer.writerow(row)
        except OSError as exc:
            log_lines.append(f"Could not save CSV file {csv_path}: {exc}")
            self.messagebox.showerror("Saving measured position", str(exc))
        try:
            with self.log_path.open("a", encoding="utf-8") as log:
                log.write("\n" + "\n".join(log_lines) + "\n")
        except OSError as exc:
            self.messagebox.showerror("Saving measured position", str(exc))
        self.status.set(f"Position saved to {csv_path.name}; {ra_text}, {dec_text}")

    @staticmethod
    def _code_for(moment: datetime) -> str:
        from comet_catalogs import position_code

def run_viewer(log_path: Path) -> None:
        return position_code(moment)

    def show_comparison(self, report: str) -> None:
        tk = self.tk
        window = tk.Toplevel(self.root)
        window.title("Comet catalog comparison")
        window.geometry("760x430")
        text = tk.Text(window, wrap="word", padx=10, pady=10)
        scroll = tk.Scrollbar(window, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        text.insert("1.0", report)
        text.configure(state="disabled")


def run_viewer(
    log_path: Path,
    mpc_catalog: Path = DEFAULT_MPC_CATALOG,
    horizons_catalog: Path = DEFAULT_HORIZONS_CATALOG,
) -> None:
    record = latest_log_record(log_path.resolve())
    viewer = CoordinateViewer(log_path.resolve(), record)
    viewer = CoordinateViewer(log_path.resolve(), record, mpc_catalog, horizons_catalog)
    viewer.root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, default=Path(__file__).resolve().with_name("astrometry_log.txt"))
    parser.add_argument("--mpc", type=Path, default=DEFAULT_MPC_CATALOG)
    parser.add_argument("--horizons", type=Path, default=DEFAULT_HORIZONS_CATALOG)
    args = parser.parse_args()
    try:
        run_viewer(args.log)
        run_viewer(args.log, args.mpc, args.horizons)
    except (RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())