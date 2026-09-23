# Examples

Start the proxy: `stuntd serve --upstream https://api.openai.com`. The upstream is the provider
origin only, without the `/v1` suffix: your client's own path is appended to it unchanged.

Set `OPENAI_API_KEY`, and `STUNTD_MODEL` if you want a model other than `gpt-4o-mini`.

Run `python examples/moderation_client.py` to send one typed decision through the proxy.

Four demos of the Jev protocol, each with its own data, client and README:

- `snake/` -- a game of Snake, one typed decision per move, trained from a BFS oracle's moves;
  the launch demo, with the numbers that run measured.
- `support/` -- triage a ticket: its category, its urgency and whether a human has to read it,
  three questions in one request.
- `banking/` -- what a customer's message is about and what to do with the payment attached to it.
- `devtools/` -- may a coding agent run this shell command, and was the last test failure a flake;
  the README shows it as a Claude Code `PreToolUse` hook.

Each demo generates its own labelled rows, imports them into a site, trains a head and then runs a
`typesafe-sdk` client against the daemon. The teacher in all four is a rule or an oracle in the
generator that reads the words of the state, so they need no API key. Five of the seven sites of the
three text demos hold a holdout agreement of 0.95 or better at the default target of 0.99; each
README has the numbers and the cost in GPU seconds.
