import type { Metadata } from 'next';

import Prose from 'components/prose';

export const metadata: Metadata = {
  title: 'Packs & Guidance',
  description: 'How evaluation, fuzzing, guidance, and advisor packs work in the Commerce ACI Benchmark.'
};

const html = `
<h1>Packs & Guidance</h1>
<p>
  Packs are opt-in, community-buildable extensions that change how runs are evaluated,
  how guidance is applied, and how fuzz cases are generated. They live under
  <code>./packs/*</code> (plus any paths in <code>COMMERCE_PACK_PATHS</code>).
</p>

<h2>What changes when you enable a pack?</h2>
<ul>
  <li><strong>Guidance packs</strong> append instruction fragments to the base system prompt. The assembled prompt and fragments are recorded in trace metadata (<code>guidance_packs</code>, <code>guidance_fragments</code>, <code>guidance_prompt_hash</code>).</li>
  <li><strong>Evaluator packs</strong> run after a trace is generated and write <code>eval_results.json</code>. Decisions are <code>pass</code>, <code>fail</code>, or <code>uncertain</code> with a severity of <code>error</code> or <code>warn</code>. A gate fails if any <code>error</code> evaluator returns <code>fail</code>. A policy can also block on <code>uncertain</code>.</li>
  <li><strong>Fuzzer packs</strong> generate multi-turn stress cases used by <code>flow_fuzz.py</code>. These do not change normal benchmark runs unless explicitly selected.</li>
  <li><strong>Advisor packs</strong> (optional LLM) propose guidance patches when evaluators fail. If enabled, a <code>guidance_patch.json</code> file is written next to the trace.</li>
</ul>

<h2>Built-in packs</h2>
<ul>
  <li><strong>commerce_safety</strong>: deterministic safety gates.
    <ul>
      <li><code>max_steps_budget</code> (error): fail if steps exceed the budget.</li>
      <li><code>loop_detector</code> (warn): detect repeated identical tool actions.</li>
      <li><code>premature_complete</code> (warn): flags TASK_COMPLETE without verifier success.</li>
    </ul>
  </li>
  <li><strong>basic_flow_fuzz</strong>: template fuzz cases (intent shift, info overload, tool injection).</li>
  <li><strong>guidance_basics</strong>: baseline guidance fragments for safer, clearer interactions.</li>
  <li><strong>advisor_guidance_patch</strong>: LLM advisor that suggests guidance patches (requires <code>ANTHROPIC_API_KEY</code>).</li>
  <li><strong>llm_judge_demo</strong>: optional LLM-based goal adherence judge (requires <code>ANTHROPIC_API_KEY</code>).</li>
</ul>

<h2>Using packs in the UI</h2>
<ul>
  <li><strong>Run Launcher</strong>: choose Guidance Packs and Eval Packs before running a task. The assembled system prompt preview shows exactly what changes.</li>
  <li><strong>Trace Viewer</strong>: run evaluators on an existing trace and view evidence. Advisors can propose a guidance patch that you can apply as a branch.</li>
</ul>

<h2>Using packs from the CLI</h2>
<pre><code>python benchmark_computeruse.py --guidance-packs guidance_basics --eval-packs commerce_safety
python flow_fuzz.py --fuzz-pack basic_flow_fuzz --fuzzer basic_strategies --turns 3,4,5
python pack_cli.py list-components --type all
python pack_cli.py eval-trace --trace debug_screenshots/.../trace.json --eval-packs commerce_safety
python pack_cli.py guidance-suggest --trace debug_screenshots/.../trace.json --advisor-pack advisor_guidance_patch --advisor suggest_patch --out guidance_patch.json
</code></pre>

<h2>Where pack documentation lives</h2>
<p>
  Each pack should include clear <code>description</code> fields in <code>pack.toml</code> for the pack
  and each component (evaluators, fuzzers, guidance, advisors). You can also add a
  <code>README.md</code> inside the pack folder for richer documentation.
</p>
`;

export default function PacksPage() {
  return <Prose className="mb-8" html={html} />;
}
