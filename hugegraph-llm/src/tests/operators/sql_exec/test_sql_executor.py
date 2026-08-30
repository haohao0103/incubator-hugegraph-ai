"""Tests for the SQL execution layer (DuckDbExecutor + runner)."""

import unittest

from hugegraph_llm.operators.sql_exec.sql_executor import (
    DEFAULT_SAMPLE_DATA,
    DuckDbExecutor,
    ExecutionResult,
)
from hugegraph_llm.operators.sql_exec.nl2sql_runner import (
    KgNL2SQLRunner,
    NL2SQLRunResponse,
)
from types import SimpleNamespace


class TestDuckDbExecutor(unittest.TestCase):
    def setUp(self):
        self.ex = DuckDbExecutor()

    def test_select_returns_columns_and_rows(self):
        r = self.ex.execute("SELECT order_id, city, amount FROM order")
        self.assertTrue(r.ok)
        self.assertEqual(r.columns, ["order_id", "city", "amount"])
        self.assertEqual(r.row_count, 8)
        self.assertEqual(len(r.rows), 8)

    def test_group_by_aggregate(self):
        r = self.ex.execute("SELECT city, SUM(amount) AS amt FROM order GROUP BY city")
        self.assertTrue(r.ok)
        self.assertEqual(r.columns, ["city", "amt"])
        self.assertEqual(r.row_count, 4)  # 北京/上海/深圳/广州
        beijing = next(row for row in r.rows if row[0] == "北京")
        self.assertAlmostEqual(beijing[1], 120.5 + 240.0 + 45.0)

    def test_join(self):
        r = self.ex.execute(
            "SELECT order.city, user.name FROM order JOIN user "
            "ON order.user_id = user.user_id WHERE order.city = '北京'"
        )
        self.assertTrue(r.ok)
        self.assertGreaterEqual(r.row_count, 3)

    def test_invalid_sql_error(self):
        r = self.ex.execute("SELECT nope FROM missing_table")
        self.assertFalse(r.ok)
        self.assertIsNotNone(r.error)
        self.assertEqual(r.columns, [])

    def test_truncation(self):
        r = self.ex.execute("SELECT * FROM order", limit=3)
        self.assertTrue(r.ok)
        self.assertEqual(len(r.rows), 3)
        self.assertEqual(r.row_count, 8)
        self.assertTrue(r.truncated)

    def test_to_dict(self):
        d = self.ex.execute("SELECT 1 AS x").to_dict()
        self.assertIn("columns", d)
        self.assertIn("duration_ms", d)

    def test_custom_sample_data(self):
        ex = DuckDbExecutor(sample_data={"t": [{"a": 1}, {"a": 2}]})
        r = ex.execute("SELECT a FROM t")
        self.assertEqual(r.row_count, 2)
        ex.close()

    def test_infer_types_and_quoting_edges(self):
        # bool column (BOOLEAN type inference), non-identifier column name
        # (skipped by the quoting pass), empty table (skipped registration),
        # and a string literal containing a keyword-ish word (masked, untouched)
        ex = DuckDbExecutor(sample_data={
            "events": [{"user id": 1, "active": True, "note": "order done"}],
            "empty_t": [],
        })
        r = ex.execute("SELECT \"user id\" FROM events WHERE note = 'order done'")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.row_count, 1)
        self.assertEqual(r.rows[0][0], 1)

    def test_close_no_raise(self):
        self.ex.close()
        self.ex.close()  # double close tolerated


class TestNl2SqlRunner(unittest.TestCase):
    class _FakePipe:
        def __init__(self, answer="SELECT 1", valid=True, question="q"):
            self._question = question
            self._answer = answer
            self._valid = valid

        def run(self, candidates=None):
            return SimpleNamespace(
                answer=self._answer,
                raw={"votes": [{"valid": self._valid}]},
                stages=[],
            )

    class _FakeExecutor:
        def __init__(self, result):
            self._result = result
            self.executed = []

        def execute(self, sql, limit=50):
            self.executed.append(sql)
            return self._result

    def _run(self, answer="SELECT 1", valid=True, result=None):
        pipe = self._FakePipe(answer=answer, valid=valid, question="订单总额")
        ex = self._FakeExecutor(result or ExecutionResult(columns=["x"], rows=[[1]], row_count=1))
        return KgNL2SQLRunner(pipe, ex).run(), ex

    def test_composes_answer_and_question(self):
        out, ex = self._run()
        self.assertIsInstance(out, NL2SQLRunResponse)
        self.assertEqual(out.question, "订单总额")
        self.assertEqual(out.sql, "SELECT 1")
        self.assertTrue(out.valid)
        self.assertIn("查询返回 1 行", out.answer)
        self.assertEqual(ex.executed, ["SELECT 1"])

    def test_no_sql_no_execution(self):
        out, ex = self._run(answer="", valid=False)
        self.assertEqual(out.sql, "")
        self.assertIsNotNone(out.execution.error)
        self.assertIn("未生成可用 SQL", out.answer)
        self.assertEqual(ex.executed, [])

    def test_invalid_sql_warns_but_runs(self):
        out, _ = self._run(valid=False)
        self.assertIn("未通过确定性校验", out.answer)

    def test_execution_error_surfaced(self):
        out, _ = self._run(
            result=ExecutionResult(error="syntax error")
        )
        self.assertFalse(out.execution.ok)
        self.assertIn("执行失败：syntax error", out.answer)

    def test_zero_rows_answer(self):
        out, _ = self._run(result=ExecutionResult(columns=["a"], rows=[], row_count=0))
        self.assertIn("查询无结果", out.answer)

    def test_truncated_answer(self):
        out, _ = self._run(
            result=ExecutionResult(
                columns=["a"], rows=[[1], [2]], row_count=99, truncated=True
            )
        )
        self.assertIn("查询返回 99 行", out.answer)
        self.assertIn("已截断", out.answer)

    def test_to_dict_roundtrip(self):
        out, _ = self._run()
        d = out.to_dict()
        self.assertEqual(d["route"], "nl2sql")
        self.assertIn("execution", d)
        self.assertIn("stages", d)
        self.assertIn("raw", d)
        self.assertEqual(d["sql"], "SELECT 1")


if __name__ == "__main__":
    unittest.main()
