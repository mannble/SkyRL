"""Trace persistence: write TraceRecord objects to JSONL files."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import IO

from skyrl_agent.meta_toolkit.observability.trace_schema import TraceRecord


class TraceWriter:
    """Thread-safe JSONL writer for TraceRecord objects.

    Writes one JSON line per trace to a file path derived from run_name and
    optional subdirectory.
    """

    def __init__(self, log_root: str | Path = "/tmp/skyrl-logs") -> None:
        self._log_root = Path(log_root)
        self._lock = Lock()
        self._handles: dict[str, IO[str]] = {}

    def _get_path(self, run_name: str, subdir: str = "meta_traces") -> Path:
        path = self._log_root / run_name / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path / "traces.jsonl"

    def write(self, record: TraceRecord, run_name: str, subdir: str = "meta_traces") -> Path:
        """Write a single TraceRecord to the JSONL file for run_name.

        Returns the path the record was written to.
        """
        file_path = self._get_path(run_name, subdir)
        line = record.model_export_json() if hasattr(record, "model_export_json") else json.dumps(
            _trace_to_dict(record), default=str
        )
        with self._lock:
            if file_path not in self._handles:
                self._handles[file_path] = open(file_path, "a", encoding="utf-8")
            self._handles[file_path].write(line + "\n")
            self._handles[file_path].flush()
        return file_path

    def write_batch(
        self, records: list[TraceRecord], run_name: str, subdir: str = "meta_traces"
    ) -> Path:
        """Write multiple TraceRecords to the JSONL file for run_name."""
        if not records:
            return self._get_path(run_name, subdir)
        file_path = self._get_path(run_name, subdir)
        with self._lock:
            handle = self._handles.get(file_path)
            if handle is None:
                handle = open(file_path, "a", encoding="utf-8")
                self._handles[file_path] = handle
            for record in records:
                line = (
                    record.model_export_json()
                    if hasattr(record, "model_export_json")
                    else json.dumps(_trace_to_dict(record), default=str)
                )
                handle.write(line + "\n")
            handle.flush()
        return file_path

    def flush_all(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                handle.flush()

    def close(self) -> None:
        with self._lock:
            for handle in self._handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _trace_to_dict(record: TraceRecord) -> dict:
    """Convert a TraceRecord to a JSON-serializable dict."""
    return {
        "task_id": record.task_id,
        "agent_version": record.agent_version,
        "policy_version": record.policy_version,
        "success": record.success,
        "final_reward": record.final_reward,
        "turn_count": record.turn_count,
        "finish_reason": record.finish_reason,
        "sync_bottleneck_score": record.sync_bottleneck_score,
        "failure_tags": record.failure_tags,
        "tool_calls": record.tool_calls,
        "tool_failures": record.tool_failures,
        "verification_attempted": record.verification_attempted,
        "retry_attempted": record.retry_attempted,
        "wall_clock_latency_ms": record.wall_clock_latency_ms,
        "idle_wait_ms": record.idle_wait_ms,
        "events": [_event_to_dict(e) for e in record.events],
        "metadata": record.metadata,
        "written_at": datetime.utcnow().isoformat(),
    }


def _event_to_dict(event) -> dict:
    """Convert a TraceEvent to a JSON-serializable dict."""
    return {
        "turn_id": event.turn_id,
        "role": event.role,
        "action_type": event.action_type,
        "content": event.content,
        "tool_name": event.tool_name,
        "tool_args": event.tool_args,
        "observation": event.observation,
        "exit_code": event.exit_code,
        "latency_ms": event.latency_ms,
        "idle_wait_ms": event.idle_wait_ms,
        "read_only_opportunity": event.read_only_opportunity,
        "fanout_opportunity": event.fanout_opportunity,
        "timeout_source": event.timeout_source,
        "tags": event.tags,
    }
