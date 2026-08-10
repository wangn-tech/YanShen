import logging
import multiprocessing
import signal
import threading

from .db import connect
from .agent_retrieval.indexing import (
    coalesce_agent_context_pending,
    finalize_agent_context_index,
    process_agent_context_pending_batch,
    rebuild_agent_context_index_if_needed,
    refresh_agent_knowledge_if_needed,
)


def _format_process_exit(exitcode):
    if exitcode is None:
        return "agent index subprocess exited without an exit code"
    if exitcode < 0:
        signal_number = -int(exitcode)
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = ""
        suffix = f" ({signal_name})" if signal_name else ""
        return f"agent index subprocess exited with signal {signal_number}{suffix}"
    return f"agent index subprocess exited with exit code {exitcode}"


def _run_agent_index_loop(
    db_path,
    stop_event,
    batch_size,
    idle_seconds,
    batch_pause_seconds,
):
    while not stop_event.is_set():
        try:
            coalesce_agent_context_pending(db_path)
            if rebuild_agent_context_index_if_needed(db_path):
                stop_event.wait(batch_pause_seconds)
                continue
            processed = process_agent_context_pending_batch(
                db_path,
                batch_size=batch_size,
            )
            if processed:
                stop_event.wait(batch_pause_seconds)
                continue
            refreshed = refresh_agent_knowledge_if_needed(db_path)
            finalized = finalize_agent_context_index(db_path)
            if refreshed or finalized:
                stop_event.wait(batch_pause_seconds)
                continue
        except Exception:
            logging.exception("Background agent index cycle failed")
        stop_event.wait(idle_seconds)


class AgentIndexWorker:
    def __init__(self, db_path, batch_size=8, idle_seconds=1.0, batch_pause_seconds=0.15):
        self.db_path = str(db_path)
        self.batch_size = max(1, int(batch_size))
        self.idle_seconds = max(0.1, float(idle_seconds))
        self.batch_pause_seconds = max(0.05, float(batch_pause_seconds))
        self._mp_context = multiprocessing.get_context("spawn")
        self._stop_event = self._mp_context.Event()
        self._thread = None
        self._process = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._supervise,
            name="gongkao-agent-indexer-supervisor",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout=5):
        self._stop_event.set()
        process = self._process
        if process and process.is_alive():
            process.join(timeout=timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=timeout)
            if process.is_alive():
                process.kill()
                process.join(timeout=timeout)
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        self._process = None

    def _start_process(self):
        process = self._mp_context.Process(
            target=_run_agent_index_loop,
            args=(
                self.db_path,
                self._stop_event,
                self.batch_size,
                self.idle_seconds,
                self.batch_pause_seconds,
            ),
            name="gongkao-agent-indexer",
        )
        process.daemon = True
        process.start()
        return process

    def _record_subprocess_exit(self, exitcode):
        message = _format_process_exit(exitcode)
        logging.error(message)
        try:
            with connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO agent_context_worker_state (id) VALUES (1)"
                )
                conn.execute(
                    """
                    UPDATE agent_context_worker_state
                       SET status = 'failed',
                           current_type = '',
                           last_error = ?,
                           updated_at = CURRENT_TIMESTAMP
                     WHERE id = 1
                    """,
                    (message[:600],),
                )
        except Exception:
            logging.exception("Failed to record agent index subprocess exit")

    def _supervise(self):
        while not self._stop_event.is_set():
            process = self._start_process()
            self._process = process
            while process.is_alive() and not self._stop_event.is_set():
                process.join(timeout=0.2)
            if self._stop_event.is_set():
                break
            exitcode = process.exitcode
            self._process = None
            if exitcode:
                self._record_subprocess_exit(exitcode)
            else:
                logging.warning("Agent index subprocess exited unexpectedly")
            self._stop_event.wait(self.idle_seconds)

    def _run(self):
        _run_agent_index_loop(
            self.db_path,
            self._stop_event,
            self.batch_size,
            self.idle_seconds,
            self.batch_pause_seconds,
        )
