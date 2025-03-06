# Dify Plugin - Gemini / with Google Gen AI SDK

forked from [Official Plugin](https://github.com/langgenius/dify-official-plugins/tree/main/models/gemini)

Modified to use the [Google Gen AI SDK](https://github.com/googleapis/python-genai) instead of the [Google AI Python SDK for the Gemini API](https://github.com/google-gemini/generative-ai-python).

## Changes

- Add `base_url` provider parameter.
- Support `Safety Settings` parameter.
- Support `Grounding`.
- Support `presence_penalty` and `frequency_penalty` parameters.
- Change `temperature` max values to 1.0 to 2.0.

---

## Other plugins

For more Dify plugins, visit: [https://github.com/blue-pen5805/dify-plugins](https://github.com/blue-pen5805/dify-plugins)
