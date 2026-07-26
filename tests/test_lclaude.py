"""Offline behavioral tests for lclaude's local launcher contract."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import lclaude


BENCH_PATH = Path(__file__).parents[1] / "lclaude-bench.py"
BENCH_SPEC = importlib.util.spec_from_file_location("lclaude_bench", BENCH_PATH)
assert BENCH_SPEC is not None and BENCH_SPEC.loader is not None
lclaude_bench = importlib.util.module_from_spec(BENCH_SPEC)
sys.modules[BENCH_SPEC.name] = lclaude_bench
BENCH_SPEC.loader.exec_module(lclaude_bench)


class FakeProcess:
    """Minimal Popen stand-in that records managed-server cleanup."""

    def __init__(self, *, returncode: int | None = None, wait_error: bool = False) -> None:
        self.returncode = returncode
        self.wait_error = wait_error
        self.terminated = 0
        self.waited = 0
        self.killed = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1

    def wait(self, timeout: int) -> None:
        self.waited += 1
        if self.wait_error:
            raise subprocess.TimeoutExpired("llama-server", timeout)

    def kill(self) -> None:
        self.killed += 1


class IsolatedPathsTestCase(unittest.TestCase):
    """Patch module-level paths so tests never touch user configuration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.settings = self.root / ".claude" / "settings.json"
        self.settings_off = self.settings.with_name("settings.json.off")
        self.config_dir = self.root / ".config" / "lclaude"
        self.config_file = self.config_dir / "config.toml"
        self.path_patches = [
            mock.patch.object(lclaude, "SETTINGS", self.settings),
            mock.patch.object(lclaude, "SETTINGS_OFF", self.settings_off),
            mock.patch.object(lclaude, "CONFIG_DIR", self.config_dir),
            mock.patch.object(lclaude, "CONFIG_FILE", self.config_file),
        ]
        for patcher in self.path_patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temp_dir.cleanup)


class SettingsLifecycleTests(IsolatedPathsTestCase):
    def _run_launcher(
        self,
        *,
        result: object | None = None,
        side_effect: object | None = None,
        server_proc: FakeProcess | None = None,
    ) -> int:
        run_kwargs = (
            {"side_effect": side_effect}
            if side_effect is not None
            else {"return_value": result}
        )
        with (
            mock.patch.object(lclaude.signal, "signal"),
            mock.patch.object(lclaude.subprocess, "run", **run_kwargs),
        ):
            return lclaude.run_claude(
                [],
                "test-model",
                backend=lclaude.BACKEND_OLLAMA,
                requested_backend=lclaude.BACKEND_OLLAMA,
                port=11434,
                server_proc=server_proc,
            )

    def test_run_claude_patches_then_restores_existing_settings(self) -> None:
        original = '{"env":{"KEEP":"yes"},"permissions":{"allow":["Read"]}}\n'
        self.settings.parent.mkdir(parents=True)
        self.settings.write_text(original)

        def assert_patch(*_args: object, **_kwargs: object) -> mock.Mock:
            patched = json.loads(self.settings.read_text())
            self.assertEqual(patched["env"]["KEEP"], "yes")
            self.assertEqual(patched["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"], "0")
            self.assertTrue(self.settings_off.exists())
            return mock.Mock(returncode=23)

        self.assertEqual(self._run_launcher(side_effect=assert_patch), 23)
        self.assertEqual(self.settings.read_text(), original)
        self.assertFalse(self.settings_off.exists())

    def test_run_claude_restores_empty_baseline_when_settings_missing(self) -> None:
        self.assertFalse(self.settings.exists())
        self.assertEqual(self._run_launcher(result=mock.Mock(returncode=0)), 0)
        self.assertEqual(self.settings.read_text(), "{}\n")
        self.assertFalse(self.settings_off.exists())

    def test_run_claude_restores_settings_and_stops_server_after_failure(self) -> None:
        original = '{"env":{"KEEP":"yes"}}\n'
        self.settings.parent.mkdir(parents=True)
        self.settings.write_text(original)
        server = FakeProcess()

        with self.assertRaisesRegex(RuntimeError, "claude unavailable"):
            self._run_launcher(
                side_effect=RuntimeError("claude unavailable"), server_proc=server
            )

        self.assertEqual(self.settings.read_text(), original)
        self.assertFalse(self.settings_off.exists())
        self.assertEqual(server.terminated, 1)
        self.assertEqual(server.waited, 1)

    def test_build_child_env_routes_locally_and_removes_cloud_key(self) -> None:
        with mock.patch.dict(
            lclaude.os.environ,
            {"ANTHROPIC_API_KEY": "cloud-key", "KEEP_THIS": "yes"},
            clear=True,
        ):
            env = lclaude.build_child_env(lclaude.BACKEND_MANAGED, 9090)

        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertEqual(env["KEEP_THIS"], "yes")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://localhost:9090")
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "lclaude")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_ALTERNATE_SCREEN"], "1")


class BackendAndCleanupTests(unittest.TestCase):
    def test_resolve_backend_prefers_usable_external_llamacpp(self) -> None:
        with (
            mock.patch.object(
                lclaude, "_is_llamacpp_reachable", return_value=(True, "build")
            ),
            mock.patch.object(
                lclaude, "_llamacpp_template_rejects_late_system_messages",
                return_value=False,
            ),
            mock.patch.object(lclaude, "check_ollama") as check_ollama,
        ):
            backend = lclaude.resolve_backend(
                "ornith:35b", ollama_port=11434, llamacpp_port=8080
            )

        self.assertEqual(backend, lclaude.BACKEND_LLAMACPP)
        check_ollama.assert_not_called()

    def test_resolve_backend_uses_managed_for_template_model(self) -> None:
        with (
            mock.patch.object(
                lclaude, "_is_llamacpp_reachable", return_value=(False, None)
            ),
            mock.patch.object(lclaude.shutil, "which", return_value="/usr/bin/llama-server"),
            mock.patch.object(lclaude, "check_ollama", return_value=(True, None)),
            mock.patch.object(lclaude, "_ollama_has_model", return_value=True),
        ):
            backend = lclaude.resolve_backend(
                "ornith:35b", ollama_port=11434, llamacpp_port=8080
            )

        self.assertEqual(backend, lclaude.BACKEND_MANAGED)

    def test_resolve_backend_falls_back_to_ollama_or_none(self) -> None:
        with (
            mock.patch.object(
                lclaude, "_is_llamacpp_reachable", return_value=(False, None)
            ),
            mock.patch.object(lclaude, "_model_needs_template_patch", return_value=False),
            mock.patch.object(lclaude, "check_ollama", return_value=(True, None)),
        ):
            self.assertEqual(
                lclaude.resolve_backend("other", ollama_port=11434, llamacpp_port=8080),
                lclaude.BACKEND_OLLAMA,
            )

        with (
            mock.patch.object(
                lclaude, "_is_llamacpp_reachable", return_value=(False, None)
            ),
            mock.patch.object(lclaude, "_model_needs_template_patch", return_value=False),
            mock.patch.object(lclaude, "check_ollama", return_value=(False, "missing")),
        ):
            self.assertIsNone(
                lclaude.resolve_backend("other", ollama_port=11434, llamacpp_port=8080)
            )

    def test_stop_process_terminates_and_kills_after_timeout(self) -> None:
        running = FakeProcess(wait_error=True)
        lclaude._stop_process(running)
        self.assertEqual((running.terminated, running.waited, running.killed), (1, 1, 1))

        stopped = FakeProcess(returncode=0)
        lclaude._stop_process(stopped)
        self.assertEqual((stopped.terminated, stopped.waited, stopped.killed), (0, 0, 0))

    def test_wait_for_llamacpp_fails_fast_when_process_has_exited(self) -> None:
        self.assertFalse(
            lclaude.wait_for_llamacpp(9090, timeout=30, proc=FakeProcess(returncode=1))
        )


class VersionStatusTests(unittest.TestCase):
    def test_version_does_not_start_or_validate_managed_backend(self) -> None:
        with (
            mock.patch.object(lclaude, "load_config", return_value={}),
            mock.patch.object(lclaude, "_print_header") as print_header,
            mock.patch.object(
                lclaude, "_get_llamacpp_binary_version", return_value="123-test"
            ),
            mock.patch.object(lclaude, "check_ollama") as check_ollama,
            mock.patch.object(lclaude, "ensure_model_in_ollama") as ensure_model,
            mock.patch.object(lclaude, "prepare_managed_backend") as prepare_managed,
            mock.patch.object(lclaude, "save_config") as save_config,
        ):
            self.assertEqual(
                lclaude.main(
                    ["--backend", lclaude.BACKEND_MANAGED, "--model", "not-pulled", "--version"]
                ),
                0,
            )

        print_header.assert_called_once_with(
            lclaude.BACKEND_MANAGED,
            lclaude.BACKEND_MANAGED,
            "123-test",
            "not-pulled",
            9090,
        )
        check_ollama.assert_not_called()
        ensure_model.assert_not_called()
        prepare_managed.assert_not_called()
        save_config.assert_not_called()

    def test_auto_version_resolves_without_starting_or_requiring_model(self) -> None:
        with (
            mock.patch.object(lclaude, "load_config", return_value={}),
            mock.patch.object(
                lclaude, "resolve_backend", return_value=lclaude.BACKEND_OLLAMA
            ) as resolve_backend,
            mock.patch.object(lclaude, "check_ollama", return_value=(False, "stopped"))
            as check_ollama,
            mock.patch.object(lclaude, "ensure_model_in_ollama") as ensure_model,
            mock.patch.object(lclaude, "_print_header"),
        ):
            self.assertEqual(lclaude.main(["--model", "not-pulled", "--version"]), 0)

        resolve_backend.assert_called_once_with(
            "not-pulled",
            ollama_port=11434,
            llamacpp_port=8080,
            auto_start=False,
            require_model=False,
        )
        check_ollama.assert_called_once_with(11434, auto_start=False)
        ensure_model.assert_not_called()


class ConfigAndHeaderTests(IsolatedPathsTestCase):
    def test_config_round_trip_escapes_model_and_omits_default_port(self) -> None:
        lclaude.save_config(model='model\\"name', backend=lclaude.BACKEND_AUTO, port=None)

        self.assertEqual(
            lclaude.load_config(),
            {"model": 'model\\"name', "backend": lclaude.BACKEND_AUTO},
        )
        self.assertNotIn("port =", self.config_file.read_text())

    def test_config_round_trip_preserves_explicit_port(self) -> None:
        lclaude.save_config(
            model="ornith:35b", backend=lclaude.BACKEND_MANAGED, port=9090
        )

        self.assertEqual(
            lclaude.load_config(),
            {
                "model": "ornith:35b",
                "backend": lclaude.BACKEND_MANAGED,
                "port": 9090,
            },
        )

    def test_invalid_config_returns_empty_mapping(self) -> None:
        self.config_dir.mkdir(parents=True)
        self.config_file.write_text("model = [\n")
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(lclaude.load_config(), {})
        self.assertIn("ignoring invalid config", stderr.getvalue())

    def test_headers_include_each_script_version(self) -> None:
        with (
            mock.patch.object(lclaude.shutil, "get_terminal_size", return_value=os.terminal_size((80, 20))),
            mock.patch("sys.stdout", new_callable=io.StringIO) as launcher_output,
        ):
            lclaude._print_header(
                lclaude.BACKEND_OLLAMA,
                lclaude.BACKEND_OLLAMA,
                "test",
                "model",
                11434,
            )
        self.assertIn(f"LCLAUDE v{lclaude.__version__}", launcher_output.getvalue())

        bench_output = io.StringIO()
        with (
            mock.patch.object(
                lclaude_bench.shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 20)),
            ),
            mock.patch.object(lclaude, "_use_color", return_value=False),
        ):
            lclaude_bench.print_bench_header(
                "model",
                (lclaude.BACKEND_OLLAMA,),
                {lclaude.BACKEND_OLLAMA: 11434},
                (lclaude_bench.Workload("quick", "Quick", "prompt", 1),),
                argparse.Namespace(
                    no_color=True,
                    unload_between=False,
                    cooldown=0,
                    warmup=0,
                    repeats=1,
                ),
                stream=bench_output,
            )
        self.assertIn(
            f"LCLAUDE-BENCH v{lclaude_bench.__version__}", bench_output.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
