# Copyright © 2026 Apple Inc.
#
# mlx-unified: image requests on an instance that serves with a draft model
# decode plainly for that request instead of being refused. Stubbed provider,
# tokenizer, and vision encoder — no checkpoint.

import unittest
from types import SimpleNamespace
from unittest import mock

from mlx_lm.server import CompletionRequest, ResponseGenerator


class StubTokenizer:
    has_chat_template = True
    has_tool_calling = False
    has_thinking = False

    def apply_chat_template(self, messages, **kwargs):
        return "rendered"


class StubEncoder:
    def prepare(self, rendered, images):
        return SimpleNamespace(tokens=[1, 2, 3], images=images)


def generator(draft_model, vision_encoder=StubEncoder()):
    gen = ResponseGenerator.__new__(ResponseGenerator)
    gen.model_provider = SimpleNamespace(
        draft_model=draft_model,
        draft_kind="mtp",
        vision_encoder=vision_encoder,
        cli_args=SimpleNamespace(chat_template_args={}),
    )
    return gen


def chat_request(content="describe this"):
    return CompletionRequest("chat", "", [{"role": "user", "content": content}], None, None)


class TestImageRequestsSkipTheDraft(unittest.TestCase):
    def test_tokenize_accepts_images_with_a_draft_loaded(self):
        gen = generator(draft_model=object())
        request = chat_request()
        args = SimpleNamespace(chat_template_kwargs=None)
        with mock.patch("mlx_lm.server.process_message_content", return_value=["img"]):
            prompt, _, _, initial_state = gen._tokenize(StubTokenizer(), request, args)
        self.assertEqual(prompt, [1, 2, 3])
        self.assertEqual(initial_state, "normal")
        self.assertIsNotNone(request.vision)

    def test_tokenize_still_refuses_images_without_vision_components(self):
        gen = generator(draft_model=None, vision_encoder=None)
        with mock.patch("mlx_lm.server.process_message_content", return_value=["img"]):
            with self.assertRaises(ValueError) as raised:
                gen._tokenize(
                    StubTokenizer(), chat_request(), SimpleNamespace(chat_template_kwargs=None)
                )
        self.assertIn("no vision components", str(raised.exception))

    def test_image_request_resolves_to_no_draft(self):
        draft = object()
        gen = generator(draft_model=draft)
        args = SimpleNamespace(speculative=True)
        text = chat_request()
        self.assertIs(gen._request_draft_model(text, args), draft)
        image = chat_request()
        image.vision = SimpleNamespace(tokens=[1, 2, 3])
        self.assertIsNone(gen._request_draft_model(image, args))

    def test_speculative_opt_out_still_wins(self):
        gen = generator(draft_model=object())
        self.assertIsNone(
            gen._request_draft_model(chat_request(), SimpleNamespace(speculative=False))
        )

    def test_no_draft_loaded_is_no_draft(self):
        gen = generator(draft_model=None)
        self.assertIsNone(gen._request_draft_model(chat_request(), SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
