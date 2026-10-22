# IFM API: terms notes

Reviewed on 9 September 2026 for this collector. This is a record of what was checked. It is not legal advice.

- Model and endpoint: `IFM/K2-Horizon-375B-A23B` at `https://api.ifm.ai/v1/chat/completions`.
- [Terms of Use](https://docs.ifm.ai/#/terms), effective 2 September 2026, prohibit: "Automatically or
  programmatically scrape Output at scale to train a competing foundation model". They permit API application
  development and assign IFM's rights in generated output, if any, to the user.
- The [Usage Policies](https://docs.ifm.ai/#/usage-policies) had no general synthetic-data prohibition at review time.
- Reading: a small tab-grouping classifier is not a competing foundation model. This is our reading, not an
  approval from IFM. Account or program terms may add conditions.
- Published [limits](https://docs.ifm.ai/#/limits): 3,000,000 prompt plus completion tokens per day, reset at
  00:00 UTC. The API's rate-limit headers reported different values during the run. The collector does not rely
  on either value: it has its own fixed lifetime token cap, starts at most one request per second, and waits for
  `Retry-After` after HTTP 429.
- Prompts contain only invented sessions. No real browsing history or personal data is sent.
- No account purchase, upgrade, top-up, key rotation or quota workaround is part of the collector.

The collector stores the SHA-256 of this file in its corpus contract.
