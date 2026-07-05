# AgentEVAL Risk Tools

AgentEVAL Risk Tools 是一个面向 LLM Agent 安全评测的上层编排原型，核心目标是把分散的攻击方法组织成统一的两阶段流程：

```text
待测 Agent -> Tool1 风险层面发现 -> Tool2 测试用例生成 -> 执行器/结果反馈
```

Tool1 负责根据待测 Agent 的访问描述、静态材料和良性动态探针发现候选风险层面，输出带证据的 Risk Seed。当前 Tool1 还可以使用受限 LLM 做两类补漏：把静态文本转成 semantic evidence，把运行时响应转成 runtime event；最终风险方向仍由确定性规则判断。Tool2 读取 Risk Seed 与 Agent Snapshot，默认走 SIRAJ-style case generation 路径，生成结构化、可校验、带 provenance 的测试用例。底层真实攻击执行器可以后续通过 executor registry 接入；当前默认使用确定性 sandbox executor 做 dry-run/proxy 验证。

## 总体架构

```mermaid
flowchart LR
    A["Agent Access Descriptor<br/>endpoint / python / runner / mock"] --> B["Direct Agent Connector"]
    B --> C["Tool1<br/>风险层面发现"]
    M["Prompt Catalog<br/>src/agenteval/prompts"] -. "受限 JSON prompt" .-> C
    C --> D["Agent Snapshot<br/>能力与证据快照"]
    C --> E["Risk Seed[]<br/>风险域 / 入口 / 前置条件 / 置信度"]
    D --> F["Tool2<br/>Seed-conditioned Case Generation"]
    E --> F
    F --> G["Generated Case[]<br/>setup / trigger / expected_signal / cleanup"]
    G --> H["Case Executor Registry"]
    H --> I["Sandbox Executor<br/>默认 dry-run/proxy"]
    H -. "后续接入" .-> J["真实底层执行器<br/>PyRIT / RAG / Memory / MCP / Planning"]
    I --> K["Run Result"]
    J --> K
    K --> L["Feedback<br/>更新 Seed 置信度与失败阶段"]
    L --> E
```

## Tool1 流程

Tool1 的定位是“证据驱动的候选风险发现器”，不是直接证明漏洞存在。它输出的是值得测试的风险假设，并保留每个判断对应的 evidence_id。

```mermaid
flowchart TD
    A["AgentAccessDescriptor"] --> B["create_connector(protocol)"]
    B --> C["handshake<br/>natural_language_input evidence"]
    B --> D["inspect<br/>capabilities / tool_schemas / policy"]
    A --> E["optional_artifacts"]
    E --> F["正则文本识别<br/>artifact_text evidence"]
    E --> G["结构化解析<br/>artifact_structured evidence"]
    E --> H["LLM semantic evidence extraction<br/>artifact_semantic evidence"]
    B --> I["benign dynamic probes"]
    I --> J["connector.send(prompt)<br/>response.content / raw / events"]
    J --> K["LLM runtime event induction<br/>补充 ConnectorEvent"]
    J --> L["connector 原生 ConnectorEvent"]
    K --> M["runtime evidence mapping"]
    L --> M
    M --> N["runtime_log evidence"]
    C --> O["EvidenceItem[]"]
    F --> O
    G --> O
    H --> O
    N --> O
    D --> P["AgentSnapshot"]
    O --> P
    P --> Q["RISK_RULES deterministic matching"]
    Q --> R["RiskSeed[]"]
    R --> S["可选 LLM Review<br/>只复核已有 Seed"]
    S --> T["SIRAJ seed enrichment<br/>risk_outcome / trajectory"]
    T --> U["risk_seeds.json"]
    P --> V["agent_snapshot.json"]
```

Tool1 中 LLM 的使用位置是受限的：

- 静态文本语义 evidence 抽取：只输出白名单 `EvidenceItem`，不直接生成 seed。
- 运行时响应事件抽取：只补充白名单 `ConnectorEvent`，再由代码映射成 runtime evidence。
- Seed review：只复核已有 seed 的证据充分性。
- SIRAJ seed enrichment：只给已有 seed 补 `risk_outcome`、`risk_source`、`expected_trajectory` 等元信息。

当前覆盖的风险域包括：

- `prompt_context_injection`
- `rag_poisoning`
- `memory_poisoning`
- `tool_output_injection`
- `mcp_description_poisoning`
- `planning_poisoning`
- `multi_agent_communication_poisoning`
- `search_narrative_poisoning`

## Tool2 流程

Tool2 的定位是“基于 Risk Seed 和 Agent Snapshot 的测试用例生成器”。它默认直接走 SIRAJ 路径：先生成模板骨架，再绑定目标上下文，然后让 SIRAJ prompt 或确定性 fallback 只改写 `setup` / `trigger` 的自然语言内容，最后做结构校验和 dry-run 校验。旧模板路径仍保留为 `--legacy-prompts` 兼容入口。

```mermaid
flowchart TD
    A["Risk Seed"] --> B["Seed Parser"]
    C["Agent Snapshot"] --> D["Context Binder<br/>工具名 / 参数 / 角色 / source_set"]
    B --> E["Strategy Selector<br/>风险域 -> 模板 -> 执行器"]
    D --> F["SIRAJ Case Synthesizer<br/>模板骨架 + SIRAJ prompt/fallback"]
    E --> F
    F --> G["Validity Checker<br/>schema / object / setup-trigger-cleanup / executor"]
    G --> H["Quality Ranker<br/>applicability / executability / goal consistency / diversity"]
    H --> I["generated_cases.json"]
```

每个 Generated Case 包含：

- `case_id`、`seed_id`、`attack_family`
- `delivery_mode`
- `setup`、`trigger`、`expected_signal`、`cleanup`
- `executor`
- `quality_score`
- `provenance`
- `validation_result`

## 目录结构

```text
src/agenteval/
  tool1/              Tool1 风险发现
  tool2/              Tool2 用例生成
  prompts/            LLM system prompt 文件目录
  connectors.py       HTTP / Python / runner / mock 连接器
  static_analysis.py  静态材料解析
  experiment.py       executor registry 与 sandbox executor
  feedback.py         结果反馈闭环
  evaluation.py       Tool1/Tool2 论文实验指标
  api.py              FastAPI 接口
  cli.py              命令行入口

examples/
  current_framework_agents.json  当前框架 agent descriptor 示例
  direct_agent_sample.json       本地 Python Agent 示例

tests/
  test_end_to_end.py             端到端测试
```

## 环境安装

推荐使用独立 conda 环境：

```powershell
conda create -n agenteval-tool12 python=3.11 -y
conda activate agenteval-tool12

Set-Location F:\Project\AgentEVAL

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install httpx "uvicorn[standard]"
```

## 快速运行

```powershell
$env:PYTHONPATH="src"
python -m agenteval.cli run-demo --out runs/demo --count 1 --no-llm-evidence --no-llm-runtime-events --no-llm-review --no-llm-variants
```

典型输出：

- `analysis_session.json`
- `agent_snapshot.json`
- `risk_seeds.json`
- `generated_cases.json`
- `run_result.json`
- `summary.json`

## 常用命令

分析单个 Agent：

```powershell
python -m agenteval.cli analyze-agent --descriptor examples/current_framework_agents.json --agent SimpleRAGChatbot --out runs/simple_rag
```

基于 Seed 生成 Case，默认走 SIRAJ 路径：

```powershell
python -m agenteval.cli generate-cases --analysis-dir runs/simple_rag --count 3
```

显式回到旧模板路径：

```powershell
python -m agenteval.cli generate-cases --analysis-dir runs/simple_rag --count 3 --legacy-prompts
```

执行 sandbox dry-run：

```powershell
python -m agenteval.cli run-cases --analysis-dir runs/simple_rag
```

把执行结果反馈到 Seed：

```powershell
python -m agenteval.cli apply-feedback --analysis-dir runs/simple_rag
```

汇总运行结果：

```powershell
python -m agenteval.cli summarize --run-root runs/demo
```

生成 Markdown 报告：

```powershell
python -m agenteval.cli write-report --run-root runs/demo --out runs/demo/report.md
```

## Prompt 目录

所有传给 `DeepSeekJSONClient.complete_json()` 的 system prompt 都集中在：

```text
src/agenteval/prompts/
```

当前 prompt 文件包括：

- `tool1_semantic_evidence_system.txt`
- `tool1_runtime_event_system.txt`
- `tool1_seed_review_system.txt`
- `tool1_siraj_enrichment_system.txt`
- `tool2_variant_system.txt`
- `tool2_siraj_case_system.txt`
- `tool2_siraj_refinement_system.txt`
- `evaluation_direct_llm_baseline_system.txt`

动态 payload 仍在 Python 代码中组装，因为它依赖当前 `AgentSnapshot`、`RiskSeed`、case skeleton 和 probe response。

## DeepSeek LLM 模式

Tool1 和 Tool2 可以选择调用 DeepSeek JSON 接口。API Key 只从环境变量读取，不应写入代码、README、JSON 输出或测试文件。

```powershell
$env:PYTHONPATH="src"
$env:DEEPSEEK_API_KEY="<your-deepseek-key>"
$env:DEEPSEEK_MODEL="deepseek-v4-pro"

python -m agenteval.cli run-demo --out runs/demo_llm --llm-evidence --llm-runtime-events --llm-review --llm-variants
```

LLM 使用边界：

- Tool1 semantic evidence 只能从静态文本中抽取白名单 feature，且必须给出原文摘录。
- Tool1 runtime event induction 只能从 probe response 中抽取白名单事件，且必须给出响应摘录。
- Tool1 seed review 只对低置信度或需要自然语言理解的已有 Seed 做结构化审查。
- Tool1 SIRAJ enrichment 只补充已有 Seed 的细粒度 outcome/source/trajectory。
- Tool1 不允许 LLM 生成没有 evidence 支撑的新风险。
- Tool2 只允许 LLM 改写 `setup` 和 `trigger` 中的自然语言内容。
- Tool2 不允许 LLM 修改 executor、工具名、expected_signal、cleanup、ID 或顶层结构。
- LLM 失败或输出非法 JSON 时，流程会记录 provenance 并回退到确定性 SIRAJ fallback。

相关 CLI 开关：

```text
--llm-evidence / --no-llm-evidence
--llm-runtime-events / --no-llm-runtime-events
--llm-review / --no-llm-review
--llm-variants / --no-llm-variants
--siraj-prompts 默认路径 / --legacy-prompts 旧模板路径
```

## FastAPI 接口

```powershell
$env:PYTHONPATH="src"
uvicorn agenteval.api:app --reload
```

接口列表：

- `POST /api/risk-discovery/analyze`
- `GET /api/analysis-sessions/{analysis_id}`
- `POST /api/case-generation/generate`
- `GET /api/generation-jobs/{job_id}`
- `POST /api/experiments/from-seeds`
- `POST /api/results/{run_id}/feedback`

接口用于后续前端接入：风险画像、Seed 证据抽屉、Case 生成任务状态、Seed -> Case -> Run 链路展示。

## Tool1/Tool2 论文实验

`evaluate-tool12` 用于生成透明的 Tool1/Tool2 功能贡献指标。标签来源必须是显式输入：可以传入 labels 文件，也可以使用 descriptor 中的 `expected_domains`。

```powershell
$env:PYTHONPATH="src"
python -m agenteval.cli evaluate-tool12 --descriptors examples/current_framework_agents.json --out runs/tool12_eval --count 1 --no-llm-evidence --no-llm-runtime-events --no-llm-review --no-llm-variants
```

输出文件：

- `tool1_metrics.json` / `tool1_metrics.csv`
- `tool2_metrics.json` / `tool2_metrics.csv`
- `baseline_metrics.json` / `baseline_metrics.csv`
- `ablation_metrics.json` / `ablation_metrics.csv`
- `evaluation_summary.json`
- `paper_tables.md`

支持的对比实验：

- `ours`
- `all_domains`
- `random_domains`
- `fixed_template`
- `direct_llm`，需要设置 `DEEPSEEK_API_KEY`

支持的消融实验：

- `ours_full`
- `w/o_static_parsing`
- `w/o_dynamic_probe`
- `w/o_semantic_evidence`
- `w/o_runtime_event_induction`
- `w/o_llm_review`
- `w/o_siraj_prompts`
- `w/o_context_binding`
- `w/o_dry_run`
- `w/o_feedback`

注意：sandbox 输出只表示 dry-run/proxy 指标，不能写成真实 ASR。真实 ASR 需要从底层真实执行器或人工整理表导入。

## 导入论文结果表

如果已有真实底层攻击结果或人工整理结果，可以用 `import-paper-results` 做格式化、汇总和 Markdown 表格生成。

```powershell
python -m agenteval.cli import-paper-results --input path\manual_results.csv --out runs/paper_tables
```

建议输入字段：

- `agent_ref`
- `method`
- `risk_domain`
- `seed_precision`
- `seed_recall`
- `schema_valid_rate`
- `dry_run_valid_rate`
- `asr`
- `source`，例如 `manual`、`real_executor`、`dry_run_proxy`

## 真实执行器接入

当前默认执行器是 deterministic sandbox。真实底层攻击执行器可以实现 `CaseExecutor` 并注册到 `ExecutorRegistry`：

```python
from agenteval.experiment import CaseExecutor, DEFAULT_EXECUTOR_REGISTRY


class RealRagExecutor(CaseExecutor):
    name = "rag_poison_runner"

    def run(self, analysis_id, cases):
        ...


DEFAULT_EXECUTOR_REGISTRY.register("rag_poison_runner", RealRagExecutor())
```

Tool2 生成的 case 会根据 `executor` 字段选择执行器。如果目标执行器没有注册，系统会回退到 sandbox，并在 result provenance 中记录 `fallback_reason`。

## 测试

```powershell
$env:PYTHONPATH="src"
$env:PYTHONDONTWRITEBYTECODE="1"
python -B -m unittest discover -s tests -v
```

当前测试覆盖：

- Tool1 风险发现与证据绑定
- Tool1 LLM semantic evidence 抽取
- Tool1 LLM runtime event induction
- Tool1 SIRAJ seed enrichment
- Tool2 case 生成、LLM 受限变体与 dry-run 校验
- Tool2 SIRAJ prompt case 生成与 refinement
- executor registry fallback
- feedback 置信度更新
- FastAPI analyze/generate/run/feedback 链路
- Tool1/Tool2 论文指标输出
- 导入显式论文结果表

## 方法边界

- Tool1 输出的是候选风险层面，不等同于已经证明漏洞存在。
- Tool2 生成的是结构化测试用例，不声称底层攻击算法全部由本工具重新发明。
- 当前 sandbox 结果不能作为真实攻击成功率。
- 真实 ASR、防御效果和业务影响指标应来自真实底层执行器或明确来源的导入表。
