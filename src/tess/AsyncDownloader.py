"""
Async Downloader - High-performance parallel file downloads using aiohttp.

Provides 5-10x speedup over synchronous requests by:
- Async I/O with connection pooling
- Large chunk sizes (1MB vs 8KB)
- Concurrent downloads with semaphore control
- Automatic retry with exponential backoff
"""

import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
import time


@dataclass
class DownloadResult:
    """Result of a download operation."""
    url: str
    filepath: Path
    success: bool
    error: Optional[str] = None
    size_bytes: int = 0
    duration_sec: float = 0.0


class AsyncDownloader:
    """
    High-performance async file downloader.

    Usage:
        downloader = AsyncDownloader(max_concurrent=10)
        results = await downloader.download_many(file_list, temp_dir)
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
        timeout: int = 300,  # 5 min per file
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Args:
            max_concurrent: Maximum simultaneous downloads
            chunk_size: Download chunk size in bytes (default 1MB)
            timeout: Timeout per file in seconds
            max_retries: Number of retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Stats
        self.total_bytes = 0
        self.total_files = 0
        self.failed_files = 0
        self.start_time = None

    async def download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        filepath: Path,
        semaphore: asyncio.Semaphore
    ) -> DownloadResult:
        """
        Download a single file with retry logic.
        """
        async with semaphore:
            start = time.time()

            for attempt in range(self.max_retries):
                size = 0  # Reset size for each attempt (fix: was accumulating on retry)
                try:
                    async with session.get(url) as response:
                        response.raise_for_status()

                        # Stream to file
                        async with aiofiles.open(filepath, 'wb') as f:
                            async for chunk in response.content.iter_chunked(self.chunk_size):
                                await f.write(chunk)
                                size += len(chunk)

                        duration = time.time() - start
                        self.total_bytes += size
                        self.total_files += 1

                        return DownloadResult(
                            url=url,
                            filepath=filepath,
                            success=True,
                            size_bytes=size,
                            duration_sec=duration
                        )

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        # Clean up partial file
                        if filepath.exists():
                            filepath.unlink()
                    else:
                        self.failed_files += 1
                        return DownloadResult(
                            url=url,
                            filepath=filepath,
                            success=False,
                            error=f"{type(e).__name__}: {str(e)}"
                        )

                except Exception as e:
                    self.failed_files += 1
                    return DownloadResult(
                        url=url,
                        filepath=filepath,
                        success=False,
                        error=f"{type(e).__name__}: {str(e)}"
                    )

            # Should not reach here, but just in case
            return DownloadResult(url=url, filepath=filepath, success=False, error="Max retries exceeded")

    async def download_many(
        self,
        files: List[Dict],
        temp_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[DownloadResult]:
        """
        Download multiple files concurrently.

        Args:
            files: List of dicts with 'url' and 'filename' keys
            temp_dir: Directory to save files
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            List of DownloadResult objects
        """
        self.start_time = time.time()
        self.total_bytes = 0
        self.total_files = 0
        self.failed_files = 0

        temp_path = Path(temp_dir)
        temp_path.mkdir(parents=True, exist_ok=True)

        semaphore = asyncio.Semaphore(self.max_concurrent)

        # Configure connection pool
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,  # Connection pool size
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,  # DNS cache TTL
            enable_cleanup_closed=True
        )

        results = []
        total = len(files)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout
        ) as session:
            tasks = []
            for file_info in files:
                url = file_info['url']
                filename = file_info['filename']
                filepath = temp_path / filename

                task = asyncio.create_task(
                    self.download_file(session, url, filepath, semaphore)
                )
                tasks.append((file_info, task))

            # Process as completed
            for i, (file_info, task) in enumerate(tasks):
                result = await task
                result.file_info = file_info  # Attach original file_info
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total)

        return results

    def get_stats(self) -> Dict:
        """Get download statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            'total_files': self.total_files,
            'failed_files': self.failed_files,
            'total_bytes': self.total_bytes,
            'total_mb': self.total_bytes / (1024 * 1024),
            'elapsed_sec': elapsed,
            'speed_mbps': (self.total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        }


def download_files_async(
    files: List[Dict],
    temp_dir: str,
    max_concurrent: int = 10,
    progress_callback: Optional[Callable] = None
) -> List[DownloadResult]:
    """
    Convenience function to download files asynchronously.

    Args:
        files: List of dicts with 'url' and 'filename' keys
        temp_dir: Directory to save files
        max_concurrent: Maximum simultaneous downloads
        progress_callback: Optional progress callback

    Returns:
        List of DownloadResult objects
    """
    downloader = AsyncDownloader(max_concurrent=max_concurrent)

    # Save current event loop (if any) to restore later
    old_loop = None
    try:
        old_loop = asyncio.get_event_loop()
        if old_loop.is_running():
            # We're inside an async context (e.g., Jupyter)
            # Use the coroutine version instead
            raise RuntimeError("Use download_files_async_coro() in async context")
    except RuntimeError:
        # No event loop exists - that's fine
        pass

    # Create new event loop for sync execution
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results = loop.run_until_complete(
            downloader.download_many(files, temp_dir, progress_callback)
        )
        return results, downloader.get_stats()
    finally:
        loop.close()
        # Restore previous event loop if it existed
        if old_loop is not None and not old_loop.is_closed():
            asyncio.set_event_loop(old_loop)


# For running in existing event loop (e.g., Jupyter)
async def download_files_async_coro(
    files: List[Dict],
    temp_dir: str,
    max_concurrent: int = 10,
    progress_callback: Optional[Callable] = None
) -> List[DownloadResult]:
    """Coroutine version for use in existing async context."""
    downloader = AsyncDownloader(max_concurrent=max_concurrent)
    results = await downloader.download_many(files, temp_dir, progress_callback)
    return results, downloader.get_stats()
