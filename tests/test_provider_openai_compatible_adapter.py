"""test_provider_openai_compatible_adapter.py - Phase 3B-0/3B-1 contract tests.

Locks the OpenAI-compatible wire dialect mapping:

  tool schema  : canonical -> function.parameters (input NOT mutated)
  response     : text / tool_calls / text+tool_calls / empty / finish_reason
  arguments    : {} / nested JSON / Chinese / escaped JSON / INVALID (fail-fast)
  usage        : prompt_tokens -> input, completion_tokens -> output,
                 prompt_tokens_details.cached_tokens -> cache_read; None -> None
  messages     : legacy wire -> OpenAI wire (role=tool for tool results),
                 system prompt as system role message
  two-round    : User -> model tool_call -> ToolCall -> ToolResultMessage
                 -> OpenAI tool wire -> model final text (golden)
  StopReason   : stop/tool_calls/length/content_filter
  exceptions   : HTTPError passthrough, invalid-arguments fail-fast
"""

import json
import types
import unittest

from agents.providers.message_codec import render_openai, to_canonical
from agents.providers.openai_compatible_adapter import (
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
    OpenAICompatibleHTTPError,
    _to_openai_tools,
)
from agents.providers.types import (
    AssistantMessage,
    ModelRequest,
    ModelResponse,
    StopReason,
    TokenUsage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


CONFIG = OpenAICompatibleConfig(
    base_url="https://api.deepseek.com/v1",
    api_key="test-key",
    model="deepseek-chat",
)

CANONICAL_TOOLS = [
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def _resp(choice_overrides=None, message_overrides=None, usage=None,
          finish="stop"):
    message = {"role": "assistant", "content": "ok"}
    if message_overrides:
        message.update(message_overrides)
    choice = {"index": 0, "message": message, "finish_reason": finish}
    if choice_overrides:
        choice.update(choice_overrides)
    body = {"id": "chatcmpl-test", "choices": [choice]}
    if usage is not None:
        body["usage"] = usage
    return body


class _FakeClient:
    def __init__(self):
        self.payloads = []
        self.response = None
        self.error = None

    def complete(self, payload):
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


class ToolSchemaConversionTests(unittest.TestCase):

    def test_canonical_to_openai_function_schema(self):
        converted = _to_openai_tools(CANONICAL_TOOLS)
        self.assertEqual(len(converted), 2)
        first = converted[0]
        self.assertEqual(first["type"], "function")
        fn = first["function"]
        self.assertEqual(fn["name"], "read_file")
        self.assertEqual(fn["description"], "Read file contents.")
        self.assertEqual(
            fn["parameters"],
            {"type": "object",
             "properties": {"path": {"type": "string"}},
             "required": ["path"]},
        )

    def test_input_schema_not_mutated(self):
        before = json.dumps(CANONICAL_TOOLS, sort_keys=True)
        _to_openai_tools(CANONICAL_TOOLS)
        after = json.dumps(CANONICAL_TOOLS, sort_keys=True)
        self.assertEqual(before, after)

    def test_none_tools(self):
        self.assertIsNone(_to_openai_tools(None))
        self.assertIsNone(_to_openai_tools([]))

    def test_description_not_lost_when_missing(self):
        converted = _to_openai_tools([
            {"name": "x", "input_schema": {"type": "object"}},
        ])
        self.assertEqual(converted[0]["function"]["description"], "")


class ResponseMappingTests(unittest.TestCase):

    def setUp(self):
        self.client = _FakeClient()
        self.adapter = OpenAICompatibleAdapter(
            CONFIG, client_provider=lambda: self.client
        )

    def _complete(self, body):
        self.client.response = body
        return self.adapter.complete(ModelRequest(
            model="deepseek-chat",
            messages=[UserMessage(content="hi")],
        ))

    def test_plain_text(self):
        resp = self._complete(_resp())
        self.assertEqual(resp.text, "ok")
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.stop_reason, StopReason.END)
        self.assertEqual(resp.provider, "openai-compatible")

    def test_single_tool_call(self):
        body = _resp(
            message_overrides={
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file",
                                 "arguments": '{"path": "x.py"}'},
                }],
            },
            finish="tool_calls",
        )
        resp = self._complete(body)
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.tool_calls, [
            ToolCall(id="call_1", name="read_file", arguments={"path": "x.py"}),
        ])
        self.assertEqual(resp.stop_reason, StopReason.TOOL_CALL)

    def test_multiple_tool_calls(self):
        body = _resp(
            message_overrides={
                "content": None,
                "tool_calls": [
                    {"id": "a", "type": "function",
                     "function": {"name": "bash",
                                  "arguments": '{"command": "ls"}'}},
                    {"id": "b", "type": "function",
                     "function": {"name": "read_file",
                                  "arguments": '{"path": "y.py"}'}},
                ],
            },
            finish="tool_calls",
        )
        resp = self._complete(body)
        self.assertEqual(
            [(tc.id, tc.name) for tc in resp.tool_calls],
            [("a", "bash"), ("b", "read_file")],
        )

    def test_text_plus_tool_calls_both_preserved(self):
        body = _resp(
            message_overrides={
                "content": "I'll check.",
                "tool_calls": [{
                    "id": "c1", "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }],
            },
            finish="tool_calls",
        )
        resp = self._complete(body)
        self.assertEqual(resp.text, "I'll check.")
        self.assertEqual(len(resp.tool_calls), 1)

    def test_empty_choices(self):
        resp = self._complete({"id": "x", "choices": []})
        self.assertEqual(resp.text, "")
        self.assertEqual(resp.tool_calls, [])
        self.assertIsNone(resp.usage)

    def test_finish_reason_mapping(self):
        cases = [
            ("stop", StopReason.END),
            ("tool_calls", StopReason.TOOL_CALL),
            ("length", StopReason.LENGTH),
            ("content_filter", StopReason.CONTENT_FILTER),
            ("weird", StopReason.UNKNOWN),
            (None, StopReason.UNKNOWN),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                resp = self._complete(_resp(finish=raw))
                self.assertEqual(resp.stop_reason, expected)

    def test_raw_finish_reason_in_metadata(self):
        resp = self._complete(_resp(finish="length"))
        self.assertEqual(resp.provider_metadata["raw_finish_reason"], "length")


class ArgumentsParsingTests(unittest.TestCase):

    def setUp(self):
        self.client = _FakeClient()
        self.adapter = OpenAICompatibleAdapter(
            CONFIG, client_provider=lambda: self.client
        )

    def _tool_call_resp(self, arguments_json):
        return _resp(
            message_overrides={
                "content": None,
                "tool_calls": [{
                    "id": "c", "type": "function",
                    "function": {"name": "f", "arguments": arguments_json},
                }],
            },
            finish="tool_calls",
        )

    def test_empty_object(self):
        self.client.response = self._tool_call_resp("{}")
        resp = self.adapter.complete(ModelRequest(model="m", messages=[]))
        self.assertEqual(resp.tool_calls[0].arguments, {})

    def test_nested_json(self):
        self.client.response = self._tool_call_resp(
            '{"a": {"b": [1, 2, {"c": null}]}}'
        )
        resp = self.adapter.complete(ModelRequest(model="m", messages=[]))
        self.assertEqual(
            resp.tool_calls[0].arguments,
            {"a": {"b": [1, 2, {"c": None}]}},
        )

    def test_chinese_content(self):
        self.client.response = self._tool_call_resp('{"content": "你好世界"}')
        resp = self.adapter.complete(ModelRequest(model="m", messages=[]))
        self.assertEqual(resp.tool_calls[0].arguments, {"content": "你好世界"})

    def test_escaped_json(self):
        self.client.response = self._tool_call_resp(
            r'{"path": "a\\b\\\"c.json"}'
        )
        resp = self.adapter.complete(ModelRequest(model="m", messages=[]))
        self.assertEqual(resp.tool_calls[0].arguments["path"], 'a\\b\\"c.json')

    def test_invalid_json_fails_fast(self):
        self.client.response = self._tool_call_resp("{not json")
        with self.assertRaises(json.JSONDecodeError):
            self.adapter.complete(ModelRequest(model="m", messages=[]))

    def test_valid_json_non_object_fails_fast(self):
        self.client.response = self._tool_call_resp('"just a string"')
        with self.assertRaises(ValueError):
            self.adapter.complete(ModelRequest(model="m", messages=[]))


class UsageMappingTests(unittest.TestCase):

    def setUp(self):
        self.client = _FakeClient()
        self.adapter = OpenAICompatibleAdapter(
            CONFIG, client_provider=lambda: self.client
        )

    def test_basic_usage(self):
        self.client.response = _resp(usage={
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "total_tokens": 168,
        })
        resp = self.adapter.complete(ModelRequest(model="m", messages=[]))
        self.assertEqual(
            resp.usage,
            TokenUsage(input_tokens=123, output_tokens=45,
                       cache_read_tokens=0, cache_write_tokens=0),
        )

    def test_cached_tokens_mapped(self):
        self.client.response = _resp(usage={
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 90},
        })
        resp = self.adapter.complete(ModelRequest(model="m", messages=[]))
        self.assertEqual(resp.usage.cache_read_tokens, 90)

    def test_usage_none(self):
        self.client.response = _resp()
        resp = self.adapter.complete(ModelRequest(model="m", messages=[]))
        self.assertIsNone(resp.usage)


class MessageWireConversionTests(unittest.TestCase):

    def test_legacy_history_to_openai_wire(self):
        """The two-round golden: user -> tool_call -> tool_result -> text."""
        legacy = [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "reading now"},
                {"type": "tool_use", "id": "t1", "name": "read_file",
                 "input": {"path": "x.py"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "hello"},
            ]},
        ]
        wire = render_openai(legacy)
        self.assertEqual(wire, [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": "reading now", "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "read_file",
                              "arguments": '{"path": "x.py"}'}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "hello"},
        ])

    def test_canonical_dataclasses_to_openai_wire(self):
        wire = render_openai([
            UserMessage(content="hi"),
            AssistantMessage(
                text="checking",
                tool_calls=[ToolCall(id="a", name="bash", arguments={"command": "ls"})],
            ),
            ToolResultMessage(tool_call_id="a", content="out"),
        ])
        self.assertEqual(wire, [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "checking", "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "bash",
                              "arguments": '{"command": "ls"}'}},
            ]},
            {"role": "tool", "tool_call_id": "a", "content": "out"},
        ])

    def test_legacy_plain_string_messages(self):
        wire = render_openai([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ])
        self.assertEqual(wire, [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ])

    def test_arguments_json_ensure_ascii_false(self):
        wire = render_openai([
            AssistantMessage(
                tool_calls=[ToolCall(id="a", name="f",
                                     arguments={"msg": "你好"})],
            ),
        ])
        self.assertIn("你好", wire[0]["tool_calls"][0]["function"]["arguments"])

    def test_to_canonical_accepts_both_shapes(self):
        canonical = to_canonical([
            {"role": "user", "content": "plain"},
            UserMessage(content="already canonical"),
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "bash",
                 "input": {"command": "ls"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
            ]},
        ])
        self.assertIsInstance(canonical[0], UserMessage)
        self.assertIsInstance(canonical[1], UserMessage)
        self.assertIsInstance(canonical[2], AssistantMessage)
        self.assertEqual(canonical[2].tool_calls[0].name, "bash")
        self.assertIsInstance(canonical[3], ToolResultMessage)
        self.assertEqual(canonical[3].tool_call_id, "t1")

    def test_tool_result_error_flag(self):
        canonical = to_canonical([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "e1",
                 "content": "Error: command not found"},
            ]},
        ])
        self.assertTrue(canonical[0].is_error)


class FullTwoRoundGoldenTests(unittest.TestCase):
    """User -> model tool_call -> harness executes -> tool wire -> final."""

    def test_two_round_golden_via_adapter(self):
        client = _FakeClient()
        adapter = OpenAICompatibleAdapter(CONFIG, client_provider=lambda: client)

        history = [{"role": "user", "content": "list files"}]
        client.response = _resp(
            message_overrides={
                "content": None,
                "tool_calls": [{
                    "id": "call_x", "type": "function",
                    "function": {"name": "bash",
                                 "arguments": '{"command": "ls"}'},
                }],
            },
            finish="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        r1 = adapter.complete(ModelRequest(model="m", messages=history,
                                          tools=CANONICAL_TOOLS))
        self.assertEqual(r1.stop_reason, StopReason.TOOL_CALL)
        self.assertEqual(r1.tool_calls[0].name, "bash")

        # Harness appends assistant wire + tool_result user message.
        history.append({
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_x", "name": "bash",
                 "input": {"command": "ls"}},
            ],
        })
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_x",
             "content": "file1  file2"},
        ]})

        client.response = _resp(
            message_overrides={"content": "Found: file1, file2"},
            usage={"prompt_tokens": 15, "completion_tokens": 8},
        )
        r2 = adapter.complete(ModelRequest(model="m", messages=history,
                                          tools=CANONICAL_TOOLS))
        self.assertEqual(r2.stop_reason, StopReason.END)
        self.assertEqual(r2.text, "Found: file1, file2")

        # The second request's wire messages must include the tool role.
        second_payload = client.payloads[1]
        roles = [m["role"] for m in second_payload["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        tool_msg = second_payload["messages"][2]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertEqual(tool_msg["tool_call_id"], "call_x")
        self.assertEqual(tool_msg["content"], "file1  file2")
        # Tools converted to OpenAI function schema in the payload.
        self.assertEqual(
            second_payload["tools"][0]["type"], "function"
        )
        self.assertEqual(
            second_payload["tools"][0]["function"]["name"], "read_file"
        )

    def test_system_prompt_as_system_role(self):
        client = _FakeClient()
        adapter = OpenAICompatibleAdapter(CONFIG, client_provider=lambda: client)
        client.response = _resp()
        adapter.complete(ModelRequest(
            model="m",
            messages=[UserMessage(content="hi")],
            system="You are a helper.",
        ))
        payload = client.payloads[0]
        self.assertEqual(payload["messages"][0],
                         {"role": "system", "content": "You are a helper."})

    def test_request_kwargs_forwarded(self):
        client = _FakeClient()
        adapter = OpenAICompatibleAdapter(CONFIG, client_provider=lambda: client)
        client.response = _resp()
        adapter.complete(ModelRequest(
            model="m",
            messages=[UserMessage(content="hi")],
            max_tokens=1234,
            temperature=0.3,
        ))
        payload = client.payloads[0]
        self.assertEqual(payload["max_tokens"], 1234)
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(payload["model"], "m")

    def test_metadata_bridge_forwards_extra_kwargs(self):
        client = _FakeClient()
        adapter = OpenAICompatibleAdapter(CONFIG, client_provider=lambda: client)
        client.response = _resp()
        adapter.complete(ModelRequest(
            model="m",
            messages=[],
            metadata={"response_format": {"type": "json_object"}},
        ))
        self.assertEqual(
            client.payloads[0]["response_format"], {"type": "json_object"}
        )

    def test_http_error_passthrough(self):
        client = _FakeClient()
        adapter = OpenAICompatibleAdapter(CONFIG, client_provider=lambda: client)
        client.error = OpenAICompatibleHTTPError("HTTP 429 ...")
        with self.assertRaises(OpenAICompatibleHTTPError):
            adapter.complete(ModelRequest(model="m", messages=[]))

    def test_unexpected_client_error_passthrough(self):
        client = _FakeClient()
        adapter = OpenAICompatibleAdapter(CONFIG, client_provider=lambda: client)
        client.error = RuntimeError("connection reset")
        with self.assertRaises(RuntimeError):
            adapter.complete(ModelRequest(model="m", messages=[]))


if __name__ == "__main__":
    unittest.main()
