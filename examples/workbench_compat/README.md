# Workbench compatibility check

Make a separate working copy of `project/` for each run. Use the same model,
workspace setup, output reserve and task in the three runs. Set
`workbench.context.strategy` to `original`, `window` or `evidence`.

Give the agent the following task:

> Add export_orders(orders) in orders.py. Start by running python3 diagnostics.py
> and use its legacy CSV contract. Then inspect and preserve the existing listing
> behavior. Add CSV export for each order in input order, supporting proper CSV
> escaping. Enter the completion flow when done.

The diagnostics command prints a compatibility requirement in the middle of
about 14 KB of setup and cleanup output. The session archives the raw output
before formatting the default observation. The project config checks Python
availability and runs the smoke check on submission. Check the actual legacy
behavior independently by running the following from mini-swe-agent:

```sh
python3 examples/workbench_compat/acceptance.py /path/to/the/working-copy
```

Use the session directory displayed when hovering over the strategy label in
the context tab. Report the provider's actual billed amount if available:

```sh
.venv/bin/mini-extra workbench-report /path/to/session --actual-cost 0.02 --currency CNY
```

The report keeps provider-reported token usage and the model adapter's cost
estimate separate from the billed amount you enter. Compare independent
acceptance, input, cache misses, and recall calls. The fixture also permits a
second diagnostic call or direct file reading, which should be counted when
comparing approaches.

The evidence strategy uses three input-budget thresholds. The soft threshold
shortens older outputs, the hard threshold removes older tool exchanges as
complete groups, and the summary threshold compresses completed user tasks if
the first two stages still leave too much context. The defaults are 0.50,
0.70 and 0.85. To change them, pass
`-c workbench.context.soft_threshold=0.45`,
`-c workbench.context.hard_threshold=0.65` or
`-c workbench.context.summary_threshold=0.80`. Keep them in ascending order.
The browser's context tab shows threshold values, summary attempts and usage.
The original and window strategies retain user task text for comparison.
