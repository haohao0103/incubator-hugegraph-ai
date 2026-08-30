"""Run the NL2SQL pipeline AND execute the winning SQL ("问 -> SQL -> 执行 -> 答案").

Wraps a :class:`KgNL2SQLPipeline` (generation + deterministic validation /
voting / lineage) with a :class:`SqlExecutor` (row-level execution) and
composes a final natural-language-ish answer from the execution result.

The pipeline itself stays execution-free; this runner is the glue that turns
"best SQL" into "answer with data", which is what an end user actually wants::

    pipe = KgNL2SQLPipeline(question="各城市订单总额", client=client)
    runner = KgNL2SQLRunner(pipe, DuckDbExecutor())
    out = runner.run()          # out.answer, out.execution, out.stages
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hugegraph_llm.operators.sql_exec.sql_executor import ExecutionResult, SqlExecutor


@dataclass
class NL2SQLRunResponse:
    """Full loop output: SQL + validation signal + executed rows + answer."""

    question: str
    sql: str
    valid: bool
    execution: ExecutionResult
    answer: str
    stages: List[Any] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route": "nl2sql",
            "question": self.question,
            "sql": self.sql,
            "valid": self.valid,
            "execution": self.execution.to_dict(),
            "answer": self.answer,
            "raw": self.raw,
            "stages": [s.model_dump() if hasattr(s, "model_dump") else s
                       for s in self.stages],
        }


class KgNL2SQLRunner:
    """Compose pipeline -> execution -> answer."""

    def __init__(self, pipeline: Any, executor: SqlExecutor) -> None:
        self._pipeline = pipeline
        self._executor = executor

    def run(self, candidates: Optional[List[str]] = None) -> NL2SQLRunResponse:
        resp = self._pipeline.run(candidates=candidates)
        sql = resp.answer or ""
        valid = bool(resp.raw.get("votes")) and bool(resp.raw["votes"][0].get("valid")) \
            if isinstance(resp.raw, dict) else False
        if not sql:
            execution = ExecutionResult(error="未生成可用 SQL（LLM 无输出或候选全为空）")
            answer = execution.error
        else:
            execution = self._executor.execute(sql)
            answer = self._compose_answer(sql, execution, valid)
        return NL2SQLRunResponse(
            question=str(getattr(self._pipeline, "_question", "")),
            sql=sql,
            valid=valid,
            execution=execution,
            answer=answer,
            stages=getattr(resp, "stages", []),
            raw=getattr(resp, "raw", {}) or {},
        )

    @staticmethod
    def _compose_answer(sql: str, execution: ExecutionResult, valid: bool) -> str:
        warn = "" if valid else "（注：该 SQL 未通过确定性校验，结果仅供参考）"
        if execution.error:
            return f"执行失败：{execution.error}{warn}"
        if execution.row_count == 0:
            return f"查询无结果（0 行）{warn}"
        cols = ", ".join(execution.columns) or "(无列)"
        head_rows = execution.rows[:3]
        head_txt = "；".join(" | ".join(str(c) for c in r) for r in head_rows)
        truncated = "（已截断，仅展示前 3 行）" if execution.truncated else ""
        return (
            f"查询返回 {execution.row_count} 行{truncated}，"
            f"字段：{cols}。示例：{head_txt}{warn}"
        )
