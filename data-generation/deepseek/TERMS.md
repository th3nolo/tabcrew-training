# DeepSeek API: terms notes

This is a record of what the collector pins. It is not legal advice.

- Model: `deepseek-v4-flash` (revision label `DeepSeek-V4-Flash-0731`).
- Endpoint: `https://api.deepseek.com/chat/completions`.
- Terms of use: https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html. The collector writes this
  link and clause 4.2(3) as the output-use permission into every exported snapshot manifest.
- Pricing: https://api-docs.deepseek.com/quick_start/pricing/. The collector estimates the cost of every
  request before sending it and reserves it against a cumulative spending cap (`--budget-usd`, at most 8 USD; the code refuses higher values).
  Unresolved billing stops collection.
- Prompts contain only invented sessions. No real browsing history or personal data is sent.
- The key is read from `DEEPSEEK_API_KEY` once and removed from the process environment.
