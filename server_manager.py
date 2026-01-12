#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Server Manager - Auto-start/stop Next.js dev servers for benchmark apps.

This module manages the lifecycle of the Next.js development servers needed
for trace replay and branch execution.

Usage:
    from server_manager import ServerManager

    manager = ServerManager(project_root=Path("."))

    # Start a server
    if manager.start_server("treatment", wait_ready=True):
        print("Server ready!")

    # Check health
    if manager.is_server_healthy("treatment"):
        print("Server is responding")

    # Stop servers on exit
    manager.stop_all()
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass
class ServerConfig:
    """Configuration for a Next.js app server."""
    app: str
    port: int
    directory: str


class ServerManager:
    """Manages Next.js dev servers for the benchmark apps."""

    CONFIGS: dict[str, ServerConfig] = {
        "baseline": ServerConfig(app="baseline", port=3001, directory="baseline"),
        "treatment": ServerConfig(app="treatment", port=3000, directory="treatment"),
        "treatment-docs": ServerConfig(app="treatment-docs", port=3002, directory="treatment-docs"),
    }

    def __init__(self, project_root: Path):
        """Initialize the server manager.

        Args:
            project_root: Root directory of the project (contains baseline/, treatment/, etc.)
        """
        self.project_root = Path(project_root)
        self._processes: dict[str, subprocess.Popen] = {}

    def get_config(self, app: str) -> ServerConfig:
        """Get configuration for an app."""
        if app not in self.CONFIGS:
            raise ValueError(f"Unknown app: {app}. Must be one of: {list(self.CONFIGS.keys())}")
        return self.CONFIGS[app]

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex(('localhost', port)) == 0

    def is_server_healthy(self, app: str) -> bool:
        """Check if server is responding to HTTP requests.

        Args:
            app: App name (baseline, treatment, treatment-docs)

        Returns:
            True if server responds with status < 500
        """
        config = self.get_config(app)
        try:
            resp = requests.get(f"http://localhost:{config.port}/", timeout=3)
            return resp.status_code < 500
        except requests.exceptions.RequestException:
            return False

    def start_server(
        self,
        app: str,
        wait_ready: bool = True,
        timeout: int = 60
    ) -> bool:
        """Start a server if not already running.

        Args:
            app: App name (baseline, treatment, treatment-docs)
            wait_ready: Whether to wait for the server to become healthy
            timeout: Maximum seconds to wait for server to be ready

        Returns:
            True if server is ready, False on timeout/error
        """
        config = self.get_config(app)

        # Check if already running and healthy
        if self.is_server_healthy(app):
            return True

        # Check if port is in use by something (maybe starting up)
        if self.is_port_in_use(config.port):
            # Port in use but not healthy - wait for it
            if wait_ready:
                return self._wait_for_healthy(app, timeout)
            return False

        # Start the server
        app_dir = self.project_root / config.directory
        if not app_dir.exists():
            raise ValueError(f"App directory not found: {app_dir}")

        env = {**os.environ, "PORT": str(config.port)}

        # Platform-specific process group handling
        kwargs = {
            "cwd": str(app_dir),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }

        if sys.platform != "win32":
            # Unix: create new process group for clean shutdown
            kwargs["preexec_fn"] = os.setsid

        try:
            proc = subprocess.Popen(
                ["npm", "run", "dev"],
                **kwargs
            )
            self._processes[app] = proc
        except FileNotFoundError:
            raise RuntimeError("npm not found. Make sure Node.js is installed.")

        if wait_ready:
            return self._wait_for_healthy(app, timeout)

        return True

    def _wait_for_healthy(self, app: str, timeout: int) -> bool:
        """Wait for server to become healthy.

        Args:
            app: App name
            timeout: Maximum seconds to wait

        Returns:
            True if healthy within timeout, False otherwise
        """
        config = self.get_config(app)
        start = time.time()

        while time.time() - start < timeout:
            if self.is_server_healthy(app):
                return True

            # Check if process died
            if app in self._processes:
                proc = self._processes[app]
                if proc.poll() is not None:
                    # Process exited
                    stdout, stderr = proc.communicate()
                    raise RuntimeError(
                        f"Server {app} exited with code {proc.returncode}.\n"
                        f"stderr: {stderr.decode('utf-8', errors='replace')[-500:]}"
                    )

            time.sleep(1)

        return False

    def stop_server(self, app: str) -> None:
        """Stop a server we started.

        Args:
            app: App name
        """
        if app not in self._processes:
            return

        proc = self._processes[app]
        if proc.poll() is None:
            # Process still running
            try:
                if sys.platform != "win32":
                    # Unix: kill process group
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    # Windows: terminate process
                    proc.terminate()

                # Wait for graceful shutdown
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if sys.platform != "win32":
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
            except (ProcessLookupError, OSError):
                # Process already dead
                pass

        del self._processes[app]

    def stop_all(self) -> None:
        """Stop all servers we started."""
        for app in list(self._processes.keys()):
            self.stop_server(app)

    def get_url(self, app: str) -> str:
        """Get the localhost URL for an app.

        Args:
            app: App name

        Returns:
            URL like "http://localhost:3000"
        """
        config = self.get_config(app)
        return f"http://localhost:{config.port}"

    def get_status(self) -> dict[str, dict]:
        """Get status of all apps.

        Returns:
            Dict mapping app name to status info
        """
        status = {}
        for app, config in self.CONFIGS.items():
            status[app] = {
                "port": config.port,
                "port_in_use": self.is_port_in_use(config.port),
                "healthy": self.is_server_healthy(app),
                "managed_by_us": app in self._processes,
            }
        return status

    @staticmethod
    def detect_app_from_trace(trace: dict) -> str:
        """Determine which app a trace was recorded against.

        Args:
            trace: Trace data dict with meta field

        Returns:
            App name: "baseline", "treatment", or "treatment-docs"
        """
        meta = trace.get("meta", {})

        # Check target_url for port
        target_url = meta.get("target_url", "")
        if ":3001" in target_url:
            return "baseline"
        elif ":3002" in target_url:
            return "treatment-docs"
        elif ":3000" in target_url:
            return "treatment"

        # Fallback to condition name parsing
        condition = meta.get("condition_name", "").lower()
        if "baseline" in condition:
            return "baseline"
        elif "docs" in condition or "treatmentdocs" in condition:
            return "treatment-docs"
        else:
            return "treatment"

    @staticmethod
    def detect_app_from_url(url: str) -> str:
        """Determine which app from a URL.

        Args:
            url: URL string

        Returns:
            App name
        """
        if ":3001" in url:
            return "baseline"
        elif ":3002" in url:
            return "treatment-docs"
        else:
            return "treatment"


# Singleton instance for convenience
_manager: Optional[ServerManager] = None


def get_server_manager(project_root: Optional[Path] = None) -> ServerManager:
    """Get or create the singleton ServerManager instance.

    Args:
        project_root: Project root directory (only used on first call)

    Returns:
        ServerManager instance
    """
    global _manager
    if _manager is None:
        if project_root is None:
            # Try to detect project root
            project_root = Path(__file__).parent
        _manager = ServerManager(project_root)
    return _manager
