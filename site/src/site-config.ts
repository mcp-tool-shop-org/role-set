import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'role-set',
  description:
    'Prototype. Plans role-shaping items, writes the parts a writer should write, records the gates each item passed, and builds kind B puzzles in code that are proven before a model words them. Nothing it writes is training data until a person checks it.',
  logoBadge: 'RS',
  brandName: 'role-set',
  repoUrl: 'https://github.com/mcp-tool-shop-org/role-set',
  footerText:
    'MIT Licensed — built by <a href="https://github.com/mcp-tool-shop-org" style="color:var(--color-muted);text-decoration:underline">mcp-tool-shop-org</a>',

  hero: {
    badge: 'Prototype · Python 3.13+ · MIT',
    headline: 'Plan every item,',
    headlineAccent: 'then write it on the loopback.',
    description:
      'role-set fixes a batch before any model call, asks a writer only for the parts a writer should write, and records the gates each item passed. Its T10 generator builds kind B puzzles in code, proves each one with the solver, and holds the variety floors with a committed record. It is a prototype. Nothing it writes is training data until a person checks it.',
    primaryCta: { href: 'https://github.com/mcp-tool-shop-org/role-set', label: 'View on GitHub' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      {
        label: 'Plan',
        code: 'uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json\n\n# slots only. no model call.\n# the same seed writes the same plan.',
      },
      {
        label: 'Mock',
        code: 'uv run role-set write --plan plan.json --pack roles/verifier --out out --mock\nuv run role-set readout --run out/mock\n\n# no model, no spend.\n# output is marked, and it is not data.',
      },
      {
        label: 'Check',
        code: 'uv run role-set puzzle-check tests/fixtures/puzzle-check/sample.jsonl\n\n# kind B only, and only the closed rule forms.\n# a free-text verdict item is outside, never passed.',
      },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'gates',
      title: 'What a run actually checks',
      subtitle: 'A refusal is a check in code, or a known-answer mismatch. Agreement from the critic is not proof.',
      features: [
        {
          title: 'Loopback only',
          desc: 'The writer and the critic must be http://127.0.0.1 with an explicit port. Any other host is refused before a byte is sent. role-set never rents a machine. A pod, when a run uses one, is offrig\'s.',
        },
        {
          title: 'The plan comes first',
          desc: 'plan fixes every slot from a seed: lesson, kind, side, and the floors. cost estimates pod hours and dollars for that plan. Both run with no model. The estimate is an estimate.',
        },
        {
          title: 'Verdict and evidence refuse',
          desc: 'On the kinds that carry them, a verdict mismatch refuses, and evidence that is not in the material refuses. Trait, answer, and flaw checks are recorded and do not refuse, until their controls pass.',
        },
        {
          title: 'The critic has not passed',
          desc: 'The trait leg measured 40.3% overall against an 85% bar, and the worst trait 12.5% against 70%. The answer leg measured 62.7% against 90%. The flaw control on 2026-10-11 had 9 of 24.',
        },
        {
          title: 'The solver is narrow',
          desc: 'Kind B covers finite assignment puzzles in a closed set of rule forms: at most 6 entities, 3 attributes, and 200,000 assignments. It does not check the thinking against the model. A verdict-keyed puzzle over free text is outside, and is not counted as passed.',
        },
        {
          title: 'The firewall is at the door',
          desc: 'ingest refuses held-out, sealed, and interview material, and DEV material during a stage. store status returns counts, not rows. Steering is designed and has never run.',
        },
      ],
    },
    {
      kind: 'data-table',
      id: 'pilots',
      title: 'Three paid pilots',
      subtitle: 'These are the only cost figures. Accept rates are R&D\'s key-check, not the runner\'s. No row from these runs is in a training set.',
      columns: ['Run', 'Date', 'What the note shows', 'Cost'],
      rows: [
        ['Pilot 1', '2026-10-11', 'None of the 104 accepted rows is data. R&D held 43 C and D rows as mis-assembled.', '$3.52'],
        ['Pilot 2', '2026-10-11', '49 candidates: 15 pass, 15 need a mechanical fix, 19 fail.', '$3.61'],
        ['Pilot 3', '2026-10-11', 'Wrote nothing. The writer\'s engine hung after startup.', '$2.80'],
      ],
    },
    {
      kind: 'code-cards',
      id: 'usage',
      title: 'Start without a model',
      cards: [
        {
          title: 'Install',
          code: '# Python 3.13+ (3.15 main), uv\nuv sync --frozen\nuv run pytest\n\n# line coverage floor: 90, enforced in CI.',
        },
        {
          title: 'A mock batch',
          code: 'uv run role-set plan --lessons T02,T03 --seed 0 --out plan.json\nuv run role-set cost --plan plan.json --profile two-card\nuv run role-set write --plan plan.json --out out --mock\nuv run role-set readout --run out/mock',
        },
      ],
    },
  ],
};
