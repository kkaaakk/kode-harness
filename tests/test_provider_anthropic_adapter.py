"""test_provider_anthropic_adapter.py - Phase 3A-1 AnthropicAdapter unit tests.

Covers the mapping contract in docs/phase3a0-provider-contract-audit.md:

  text block -> ModelResponse.text
  tool_use -> ToolCall
  multiple tool_use
  text + tool mixed
  stop_reason mapping (all 5)
  usage mapping (input/output + cache fallback)
  usage=None
  empty content
  unknown content block (preserved, not dropped)
  raw_response retained
  exception passthrough (no wrapping)
  request kwargs: exact forwarding incl. max_tokens/temperature/tools;
  metadata bridge forwards extra kwargs verbatim (extension patch compat)
"""

import types
import unittest
from dataclasses import dataclass, field
from typing import Any

from agents.providers.anthropic_adapter import AnthropicAdapter
from agents.providers.types import (
    ModelRequest,
    ModelResponse,
    StopReason,
    ToolCall,
    TokenUsage,
)


class _Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class _ToolUse:
    type = "tool_use"

    def __init__(self, name, tool_id, input_data):
        self.name = name
        self.id = tool_id
        self.input = input_data


class _Usage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_read=None,
                 cache_write=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if cache_read is not None:
            self.cache_read_input_tokens = cache_read
        if cache_write is not None:
            self.cache_creation_input_tokens = cache_write


class _Resp:
    def __init__(self, content, stop_reason="end_turn", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.response = None
        self.error = None
        self.messages = types.SimpleNamespace(create=self.messages_create)

    def messages_create(self, **kwargs):
        if self.error is not None:
            raise self.error
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class AnthropicAdapterTests(unittest.TestCase):

    def setUp(self):
        self.client = _FakeClient()
        self.adapter = AnthropicAdapter(self.client)

    def _complete(self, response=None, request=None):
        if response is not None:
            self.client.response = response
        req = request or ModelRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        return self.adapter.complete(req)

    # ------------------------------------------------------------------
    # text block
    # ------------------------------------------------------------------

    def test_text_block_maps_to_response_text(self):
        resp = self._complete(_Resp([_Text("hello world")], "end_turn"))
        self.assertEqual(resp.text, "hello world")
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.provider, "anthropic")
        self.assertEqual(resp.model, "test-model")

    def test_multiple_text_blocks_concatenate_in_order(self):
        resp = self._complete(_Resp([_Text("a"), _Text("b")], "end_turn"))
        self.assertEqual(resp.text, "ab")

    # ------------------------------------------------------------------
    # tool use
    # ------------------------------------------------------------------

    def test_tool_use_maps_to_tool_call(self):
        resp = self._complete(_Resp(
            [_ToolUse("bash", "toolu_1", {"command": "ls"})], "tool_use"
        ))
        self.assertEqual(resp.tool_calls, [
            ToolCall(id="toolu_1", name="bash", arguments={"command": "ls"}),
        ])
        self.assertEqual(resp.text, "")

    def test_multiple_tool_calls(self):
        resp = self._complete(_Resp([
            _ToolUse("read_file", "a", {"path": "x.py"}),
            _ToolUse("bash", "b", {"command": "echo hi"}),
        ], "tool_use"))
        self.assertEqual(
            [(tc.id, tc.name) for tc in resp.tool_calls],
            [("a", "read_file"), ("b", "bash")],
        )

    def test_text_plus_tool_call_both_preserved(self):
        resp = self._complete(_Resp(
            [_Text("checking..."), _ToolUse("read_file", "t1", {"path": "m.py"})],
            "tool_use",
        ))
        self.assertEqual(resp.text, "checking...")
        self.assertEqual(len(resp.tool_calls), 1)

    # ------------------------------------------------------------------
    # stop reason mapping
    # ------------------------------------------------------------------

    def test_stop_reason_mapping(self):
        cases = [
            ("end_turn", StopReason.END),
            ("tool_use", StopReason.TOOL_CALL),
            ("max_tokens", StopReason.LENGTH),
            ("stop_sequence", StopReason.STOP_SEQUENCE),
            ("weird_thing", StopReason.UNKNOWN),
            (None, StopReason.UNKNOWN),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                resp = self._complete(_Resp([_Text("x")], raw))
                self.assertEqual(resp.stop_reason, expected)

    def test_raw_stop_reason_preserved_in_metadata(self):
        resp = self._complete(_Resp([_Text("x")], "max_tokens"))
        self.assertEqual(resp.provider_metadata["raw_stop_reason"], "max_tokens")

    # ------------------------------------------------------------------
    # usage mapping
    # ------------------------------------------------------------------

    def test_usage_input_output(self):
        resp = self._complete(_Resp(
            [_Text("x")], "end_turn", _Usage(input_tokens=11, output_tokens=3)
        ))
        self.assertEqual(
            resp.usage,
            TokenUsage(input_tokens=11, output_tokens=3,
                       cache_read_tokens=0, cache_write_tokens=0),
        )

    def test_usage_cache_tokens_mapped_with_fallback(self):
        resp = self._complete(_Resp(
            [_Text("x")], "end_turn",
            _Usage(input_tokens=1, output_tokens=2,
                   cache_read=300, cache_write=50),
        ))
        self.assertEqual(
            resp.usage,
            TokenUsage(input_tokens=1, output_tokens=2,
                       cache_read_tokens=300, cache_write_tokens=50),
        )

    def test_usage_none(self):
        resp = self._complete(_Resp([_Text("x")], "end_turn", usage=None))
        self.assertIsNone(resp.usage)

    # ------------------------------------------------------------------
    # edge content
    # ------------------------------------------------------------------

    def test_empty_content(self):
        resp = self._complete(_Resp([], "end_turn"))
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.tool_calls, [])

    def test_unknown_block_preserved_not_dropped(self):
        class ThinkingBlock:
            type = "thinking"
            thinking = "secret reasoning"

        block = ThinkingBlock()
        resp = self._complete(_Resp([block], "end_turn"))
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.provider_metadata["unknown_blocks"], [block])

    # ------------------------------------------------------------------
    # raw_response retained (D-2)
    # ------------------------------------------------------------------

    def test_raw_response_retained(self):
        raw = _Resp([_Text("x")], "end_turn")
        resp = self._complete(raw)
        self.assertIs(resp.raw_response, raw)

    # ------------------------------------------------------------------
    # exception passthrough (D-3)
    # ------------------------------------------------------------------

    def test_exception_passthrough(self):
        class BoomError(Exception):
            pass

        self.client.error = BoomError("api down")
        with self.assertRaises(BoomError):
            self._complete()

    def test_usage_attribute_missing_on_usage_object(self):
        # Some SDK versions omit cache fields entirely.
        class MinimalUsage:
            input_tokens = 5
            output_tokens = 6

        resp = self._complete(_Resp([_Text("x")], "end_turn", MinimalUsage()))
        self.assertEqual(
            resp.usage,
            TokenUsage(input_tokens=5, output_tokens=6,
                       cache_read_tokens=0, cache_write_tokens=0),
        )

    # ------------------------------------------------------------------
    # request forwarding
    # ------------------------------------------------------------------

    def test_request_kwargs_forwarded_exactly(self):
        self.client.response = _Resp([_Text("ok")], "end_turn")
        self.adapter.complete(ModelRequest(
            model="m1",
            messages=[{"role": "user", "content": "u"}],
            tools=[{"name": "bash", "input_schema": {}}],
            system="sys",
            max_tokens=8000,
            temperature=0.2,
        ))
        self.assertEqual(len(self.client.calls), 1)
        kwargs = self.client.calls[0]
        self.assertEqual(kwargs["model"], "m1")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "u"}])
        self.assertEqual(kwargs["tools"], [{"name": "bash", "input_schema": {}}])
        self.assertEqual(kwargs["system"], "sys")
        self.assertEqual(kwargs["max_tokens"], 8000)
        self.assertEqual(kwargs["temperature"], 0.2)

    def test_request_none_fields_omitted(self):
        self.client.response = _Resp([_Text("ok")], "end_turn")
        self.adapter.complete(ModelRequest(
            model="m1",
            messages=[{"role": "user", "content": "u"}],
        ))
        kwargs = self.client.calls[0]
        self.assertNotIn("tools", kwargs)
        self.assertNotIn("system", kwargs)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)

    def test_metadata_bridge_forwards_extra_kwargs(self):
        """Extension patches may add request kwargs with no unified field;
        they must reach the client verbatim (3A bridge)."""
        self.client.response = _Resp([_Text("ok")], "end_turn")
        self.adapter.complete(ModelRequest(
            model="m1",
            messages=[],
            metadata={"stop_sequences": ["</answer>"]},
        ))
        kwargs = self.client.calls[0]
        self.assertEqual(kwargs["stop_sequences"], ["</answer>"])


if __name__ == "__main__":
    unittest.main()
