"""
Rich Live Dashboard for TESS Streaming Pipeline.

Provides a dynamic, visually appealing progress display using the rich library.
Falls back to the plain ProgressDisplay if rich is not available.
"""

import threading
import time
import shutil
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
from rich.columns import Columns
from rich.text import Text
from rich.layout import Layout
from rich import box


# Unicode block characters for sparkline (8 levels)
_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values, width=24):
    """Build a sparkline string from a list of numeric values."""
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    rng = hi - lo if hi > lo else 1.0
    chars = []
    # Take only the last `width` values
    for v in list(values)[-width:]:
        idx = int((v - lo) / rng * 7)
        idx = max(0, min(7, idx))
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def _format_duration(seconds):
    """Format seconds into Xh Ym Zs string."""
    if seconds < 0:
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


class RichProgressDisplay:
    """
    Rich-based live dashboard for the streaming pipeline.

    Same public interface as the plain ProgressDisplay so it can be
    used as a drop-in replacement.
    """

    def __init__(self, sector: int, camera: str, ccd: str,
                 total_files: int, already_processed: int):
        self.sector = sector
        self.camera = camera
        self.ccd = ccd
        self.total_files = total_files
        self.already_processed = already_processed
        self.remaining = total_files - already_processed

        # Counters
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.stars_detected = 0
        self.measurements = 0
        self.start_time = None
        self.errors = []

        # Speed tracking (samples every ~15 seconds for sparkline)
        self._speed_history = deque(maxlen=30)
        self._speed_sample_time = None
        self._speed_sample_count = 0

        # Recent files log
        self._recent_files = deque(maxlen=5)

        # Thread safety
        self._lock = threading.Lock()

        # Rich objects
        self._console = Console()
        self._live = None
        self._progress = None
        self._task_id = None

        # Debounce: max ~4 Hz refresh
        self._last_render = 0.0
        self._render_interval = 0.25

    def show_header(self):
        """Display initial parameters as a Rich panel (printed once before Live starts)."""
        info_lines = [
            f"[bold cyan]Sector:[/] {self.sector}  │  "
            f"[bold cyan]Camera:[/] {self.camera}  │  "
            f"[bold cyan]CCD:[/] {self.ccd}",
            "",
            f"  Total files:    [bold]{self.total_files:,}[/]",
            f"  Already done:   [green]{self.already_processed:,}[/]",
            f"  To process:     [yellow]{self.remaining:,}[/]",
        ]
        panel = Panel(
            "\n".join(info_lines),
            title="[bold white]TESS Streaming Pipeline[/]",
            border_style="blue",
            padding=(1, 2),
        )
        self._console.print(panel)

    def start(self):
        """Start the live dashboard."""
        self.start_time = datetime.now()
        self._speed_sample_time = time.monotonic()
        self._speed_sample_count = 0

        # Create the Rich Progress bar
        self._progress = Progress(
            TextColumn("[bold blue]Overall"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[bold]{task.percentage:>5.1f}%"),
            TimeElapsedColumn(),
            console=self._console,
            expand=False,
        )
        self._task_id = self._progress.add_task("Processing", total=self.remaining)

        # Start Live context
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def update(self, success: bool, stars: int = 0, measurements: int = 0,
               error: str = None, filename: str = None, date_obs: str = None,
               **kwargs):
        """Update stats after processing a file (thread-safe)."""
        with self._lock:
            self.processed += 1
            if success:
                self.successful += 1
                self.stars_detected = max(self.stars_detected, stars)
                self.measurements += measurements
            else:
                self.failed += 1
                if error:
                    self.errors.append(error)

            # Update progress bar
            if self._progress and self._task_id is not None:
                self._progress.advance(self._task_id)

            # Track recent file
            short_name = self._shorten_filename(filename) if filename else "—"
            if success:
                self._recent_files.append(("OK", short_name, f"stars={stars:,}"))
            else:
                err_short = (error or "unknown")[:40]
                self._recent_files.append(("ERR", short_name, err_short))

            # Speed sampling (~every 15 seconds)
            now = time.monotonic()
            self._speed_sample_count += 1
            if self._speed_sample_time is not None:
                dt = now - self._speed_sample_time
                if dt >= 15.0:
                    rate = self._speed_sample_count / dt * 60  # files/min
                    self._speed_history.append(rate)
                    self._speed_sample_time = now
                    self._speed_sample_count = 0

            # Debounced render
            if self._live and (now - self._last_render) >= self._render_interval:
                self._last_render = now
                self._live.update(self._build_layout())

    def close(self):
        """Stop the live dashboard."""
        # Final render
        if self._live:
            try:
                self._live.update(self._build_layout())
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def show_summary(self, output_dir: Path, catalog_file: str,
                     data_file: str, data_size_mb: float):
        """Display final summary as a Rich panel."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        # Status header
        if self.failed == 0:
            status = "[bold green]COMPLETED SUCCESSFULLY[/]"
        elif self.failed < self.processed * 0.1:
            status = "[bold yellow]COMPLETED (minor errors)[/]"
        else:
            status = "[bold red]COMPLETED WITH ERRORS[/]"

        # Build results table
        results = Table(show_header=False, box=None, padding=(0, 2))
        results.add_column("label", style="dim")
        results.add_column("value", style="bold")
        results.add_row("Stars detected", f"{self.stars_detected:,}")
        results.add_row("Epochs processed", f"{self.successful:,}")
        results.add_row("Total measurements", f"{self.measurements:,}")
        results.add_row("Failed", f"{self.failed:,}")
        results.add_row("Total time", _format_duration(elapsed))
        if self.successful > 0 and elapsed > 0:
            rate = self.successful / elapsed * 60
            results.add_row("Speed", f"{rate:.1f} files/min")

        # Output files
        files_text = (
            f"  {output_dir}/\n"
            f"    ├─ {catalog_file}\n"
            f"    └─ {data_file} ({data_size_mb:.1f} MB)"
        )

        content = Table.grid(padding=(1, 0))
        content.add_row(Text.from_markup(status, justify="center"))
        content.add_row(results)
        content.add_row(Text("OUTPUT FILES", style="bold"))
        content.add_row(Text(files_text))

        panel = Panel(content, title="[bold]Summary[/]", border_style="green", padding=(1, 2))
        self._console.print()
        self._console.print(panel)

        if self.errors:
            self._console.print(
                f"\n  [yellow]Warning:[/] {len(self.errors)} files had errors (see log for details)"
            )

    # --- Internal helpers ---

    def _build_layout(self):
        """Assemble the full dashboard layout."""
        grid = Table.grid(expand=True)

        # Row 1: progress bar
        grid.add_row(self._progress)

        # Row 2: stats + timing side by side
        grid.add_row(self._build_two_column())

        # Row 3: recent files
        grid.add_row(self._build_recent_files())

        return Panel(
            grid,
            title="[bold white]TESS Streaming Pipeline[/]",
            subtitle=f"Sector {self.sector} | Cam {self.camera} | CCD {self.ccd}",
            border_style="blue",
            padding=(0, 1),
        )

    def _build_two_column(self):
        """Build stats + timing/speed side by side."""
        left = self._build_stats_table()
        right = self._build_timing_table()

        cols = Table.grid(expand=True)
        cols.add_column(ratio=1)
        cols.add_column(ratio=1)
        cols.add_row(left, right)
        return cols

    def _build_stats_table(self):
        """Stars, epochs, measurements, failed, speed."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        speed = self.processed / elapsed * 60 if elapsed > 0 and self.processed > 0 else 0.0

        tbl = Table(show_header=True, header_style="bold", box=box.SIMPLE, padding=(0, 1))
        tbl.add_column("Statistic", style="dim", min_width=14)
        tbl.add_column("Value", justify="right")

        tbl.add_row("Stars", f"[bold]{self.stars_detected:,}[/]")
        tbl.add_row("Epochs", f"{self.successful:,}")
        tbl.add_row("Measurements", f"{self.measurements:,}")
        tbl.add_row("Failed", f"[red]{self.failed}[/]" if self.failed else "0")
        tbl.add_row("Speed", f"{speed:.1f} f/min")
        return tbl

    def _build_timing_table(self):
        """Elapsed, ETA, sparkline."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        # ETA
        if self.processed > 0 and elapsed > 0:
            files_left = self.remaining - self.processed
            eta_sec = (elapsed / self.processed) * files_left
        else:
            eta_sec = -1

        tbl = Table(show_header=True, header_style="bold", box=box.SIMPLE, padding=(0, 1))
        tbl.add_column("Timing", style="dim", min_width=14)
        tbl.add_column("Value", justify="right")

        tbl.add_row("Elapsed", _format_duration(elapsed))
        tbl.add_row("ETA", _format_duration(eta_sec))

        # Sparkline row
        spark = _sparkline(self._speed_history)
        if spark:
            avg = sum(self._speed_history) / len(self._speed_history)
            latest = self._speed_history[-1] if self._speed_history else 0
            tbl.add_row("Speed (f/min)", f"{spark}  avg:{avg:.0f} now:{latest:.0f}")
        else:
            tbl.add_row("Speed (f/min)", "[dim]collecting...[/]")

        return tbl

    def _build_recent_files(self):
        """Show last N processed files."""
        tbl = Table(
            show_header=True, header_style="bold dim",
            box=box.SIMPLE, padding=(0, 1), expand=True,
        )
        tbl.add_column("", width=4)
        tbl.add_column("File", ratio=3)
        tbl.add_column("Info", ratio=1)

        if not self._recent_files:
            tbl.add_row("", "[dim]waiting for first file...[/]", "")
        else:
            for status, fname, info in self._recent_files:
                if status == "OK":
                    tag = "[green]OK[/]"
                else:
                    tag = "[red]ERR[/]"
                tbl.add_row(tag, fname, info)

        return Panel(tbl, title="Recent Files", border_style="dim", padding=(0, 0))

    @staticmethod
    def _shorten_filename(filename):
        """Shorten TESS filename for display (keep date portion)."""
        if not filename:
            return "—"
        # TESS filenames like: tess2024123456789-s0070-1-1-0266-s_ffic.fits
        # Show: tess2024...ffic.fits
        name = str(filename)
        if len(name) > 45:
            return name[:20] + "..." + name[-15:]
        return name
