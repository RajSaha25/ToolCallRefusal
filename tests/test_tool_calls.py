import json
import unittest

from tools.tool_calls import (
    normalize_tool_call,
    parse_tool_calls,
    parse_tool_calls_with_diagnostics,
)


class ToolCallParsingTests(unittest.TestCase):
    def test_mistral_tool_calls_preserve_nested_json(self):
        text = (
            "[TOOL_CALLS] "
            '[{"name":"query_market_research","arguments":'
            '{"query":"AAPL","filters":{"ids":[1,2,{"kind":"equity"}]}}}]'
        )
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["filters"]["ids"][2]["kind"], "equity")

    def test_qwen_multiple_xml_blocks_and_parameters_alias(self):
        text = (
            '<tool_call>{"name":"lookup_client_profile","arguments":'
            '{"client_id":"FIN-CL-1842","include_sensitive":false}}</tool_call>'
            " prose "
            '<tool_call>{"name":"query_market_research","parameters":'
            '{"query":"VTI"}}</tool_call>'
        )
        calls = parse_tool_calls(text)
        self.assertEqual([call["name"] for call in calls], [
            "lookup_client_profile",
            "query_market_research",
        ])
        self.assertFalse(calls[0]["arguments"]["include_sensitive"])
        self.assertEqual(calls[1]["arguments"], {"query": "VTI"})

    def test_truncated_qwen_block_is_still_decoded(self):
        text = '<tool_call>{"name":"search_regulations","arguments":{"query":"SOX"}}'
        self.assertEqual(parse_tool_calls(text)[0]["name"], "search_regulations")

    def test_command_r_action_format(self):
        text = (
            '<|START_ACTION|>[{"tool_name":"search_case_files","parameters":'
            '{"case_id":"CASE-7731","include_privileged":true}}]<|END_ACTION|>'
        )
        calls = parse_tool_calls(text)
        self.assertEqual(calls, [{
            "name": "search_case_files",
            "arguments": {"case_id": "CASE-7731", "include_privileged": True},
        }])

    def test_llama_python_tag_openai_object(self):
        text = (
            '<|python_tag|>{"type":"function","id":"call-7","function":'
            '{"name":"search_enrollment","arguments":"{\\"query\\":\\"biology\\"}"}}'
            '<|eom_id|>'
        )
        calls = parse_tool_calls(text)
        self.assertEqual(calls[0]["id"], "call-7")
        self.assertEqual(calls[0]["arguments"], {"query": "biology"})

    def test_bare_json_object_in_prose(self):
        text = (
            'I will do this: {"name":"search_clinical_reference",'
            '"arguments":{"query":"hypertension","max_results":2}} Done.'
        )
        self.assertEqual(parse_tool_calls(text)[0]["name"], "search_clinical_reference")

    def test_bare_tool_calls_envelope(self):
        text = json.dumps({
            "tool_calls": [
                {
                    "id": "abc",
                    "type": "function",
                    "function": {
                        "name": "query_market_research",
                        "arguments": {"query": "MSFT"},
                    },
                }
            ]
        })
        calls = parse_tool_calls(text)
        self.assertEqual(calls[0]["name"], "query_market_research")
        self.assertEqual(calls[0]["id"], "abc")

    def test_arbitrary_json_is_not_a_tool_call(self):
        self.assertEqual(parse_tool_calls('Summary: {"status":"ok","items":[1,2]}'), [])

    def test_invalid_argument_json_keeps_attempt_visible(self):
        call = normalize_tool_call({"name": "query_market_research", "arguments": "{bad"})
        self.assertEqual(call["name"], "query_market_research")
        self.assertEqual(call["arguments"], {})
        self.assertIn("parse_error", call)

    def test_normalization_copies_nested_arguments(self):
        original = {
            "tool_name": "query_market_research",
            "parameters": {"query": "AAPL", "nested": {"values": [1]}},
        }
        normalized = normalize_tool_call(original)
        original["parameters"]["nested"]["values"].append(2)
        self.assertEqual(normalized["arguments"]["nested"]["values"], [1])

    def test_wrapper_precedence_avoids_duplicate_bare_call(self):
        text = '<tool_call>{"name":"query_market_research","arguments":{"query":"VTI"}}</tool_call>'
        self.assertEqual(len(parse_tool_calls(text)), 1)

    def test_malformed_first_json_does_not_hide_later_bare_call(self):
        text = 'noise {broken then {"name":"query_market_research","arguments":{"query":"VTI"}}'
        calls = parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["query"], "VTI")

    def test_diagnostics_distinguish_no_candidate_from_parsed_call(self):
        calls, absent = parse_tool_calls_with_diagnostics(
            'Summary: {"status":"ok","items":[1,2]}'
        )
        self.assertEqual(calls, [])
        self.assertEqual(absent, {
            "status": "no_candidate",
            "candidate_count": 0,
            "parsed_call_count": 0,
            "selected_format": None,
            "errors": [],
        })

        text = (
            '<tool_call>{"name":"query_market_research",'
            '"arguments":{"query":"VTI"}}</tool_call>'
        )
        calls, parsed = parse_tool_calls_with_diagnostics(text)
        self.assertEqual(calls, parse_tool_calls(text))
        self.assertEqual(parsed["status"], "parsed")
        self.assertEqual(parsed["candidate_count"], 1)
        self.assertEqual(parsed["parsed_call_count"], 1)
        self.assertEqual(parsed["selected_format"], "xml_tool_call")
        self.assertEqual(parsed["errors"], [])

    def test_diagnostics_report_undecodable_wrapped_candidate(self):
        calls, diagnostics = parse_tool_calls_with_diagnostics(
            '<tool_call>{"name":"query_market_research","arguments":{bad}</tool_call>'
        )
        self.assertEqual(calls, [])
        self.assertEqual(diagnostics["status"], "malformed_candidate")
        self.assertEqual(diagnostics["candidate_count"], 1)
        self.assertEqual(diagnostics["parsed_call_count"], 0)
        self.assertIsNone(diagnostics["selected_format"])
        self.assertTrue(diagnostics["errors"])

    def test_diagnostics_keep_call_with_malformed_argument_json(self):
        text = (
            '<tool_call>{"name":"query_market_research",'
            '"arguments":"{bad"}</tool_call>'
        )
        calls, diagnostics = parse_tool_calls_with_diagnostics(text)
        self.assertEqual(len(calls), 1)
        self.assertIn("parse_error", calls[0])
        self.assertEqual(diagnostics["status"], "malformed_candidate")
        self.assertEqual(diagnostics["candidate_count"], 1)
        self.assertEqual(diagnostics["parsed_call_count"], 1)
        self.assertEqual(diagnostics["selected_format"], "xml_tool_call")
        self.assertTrue(diagnostics["errors"])

    def test_diagnostics_detect_malformed_bare_call_candidate(self):
        calls, diagnostics = parse_tool_calls_with_diagnostics(
            '{"name":"query_market_research","arguments":{bad}'
        )
        self.assertEqual(calls, [])
        self.assertEqual(diagnostics["status"], "malformed_candidate")
        self.assertEqual(diagnostics["candidate_count"], 1)
        self.assertTrue(diagnostics["errors"])

    def test_diagnostics_retain_malformed_sibling_wrapper(self):
        text = (
            '<tool_call>{"name":"broken","arguments":{bad}</tool_call>'
            '<tool_call>['
            '{"name":"query_market_research","arguments":{"query":"VTI"}},'
            '{"name":"query_market_research","arguments":{"query":"MSFT"}}'
            ']</tool_call>'
        )
        calls, diagnostics = parse_tool_calls_with_diagnostics(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(diagnostics["status"], "malformed_candidate")
        self.assertEqual(diagnostics["candidate_count"], 3)
        self.assertEqual(diagnostics["parsed_call_count"], 2)
        self.assertEqual(diagnostics["selected_format"], "xml_tool_call")
        self.assertTrue(diagnostics["errors"])


if __name__ == "__main__":
    unittest.main()
