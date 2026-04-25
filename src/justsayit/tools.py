"""Custom tool definitions — loaded from tools.json, executed on LLM tool calls."""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    exec: str = ""

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def load_tools(path: str | Path) -> list[ToolDefinition]:
    p = Path(path).expanduser()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("could not load tools from %s: %s", p, exc)
        return []
    if not isinstance(raw, list):
        log.warning("tools.json must be a JSON array; got %s — ignoring", type(raw).__name__)
        return []
    tools: list[ToolDefinition] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        description = item.get("description", "")
        parameters = item.get("parameters", {"type": "object", "properties": {}})
        exec_cmd = item.get("exec", "")
        if not name:
            log.warning("skipping tool with missing 'name' field")
            continue
        tools.append(ToolDefinition(name=name, description=description, parameters=parameters, exec=exec_cmd))
    log.info("loaded %d tool(s) from %s", len(tools), p)
    return tools


def execute_tool(tool: ToolDefinition, params: dict) -> str:
    """Execute *tool* with *params* by running its ``exec`` shell template.

    ``{param_name}`` placeholders in the exec string are substituted with
    the corresponding parameter values. Returns stdout on success or a brief
    error message on failure.
    """
    if not tool.exec:
        return f"Tool '{tool.name}' has no exec command configured."
    cmd = tool.exec
    for key, value in params.items():
        cmd = cmd.replace("{" + key + "}", str(value))
    log.info("executing tool %r: %s", tool.name, cmd)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
    except subprocess.TimeoutExpired:
        log.warning("tool %r timed out", tool.name)
        return f"Tool '{tool.name}' timed out."
    except Exception as exc:
        log.exception("tool %r exec failed", tool.name)
        return f"Tool '{tool.name}' failed: {exc}"
    output = proc.stdout.strip()
    if proc.returncode != 0:
        err = proc.stderr.strip()
        log.warning("tool %r exited with %d: %s", tool.name, proc.returncode, err)
        return output or err or f"Tool '{tool.name}' exited with code {proc.returncode}."
    return output or "(done)"
