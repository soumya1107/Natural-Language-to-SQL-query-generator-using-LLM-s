import unittest

from core.prompt_builder import PromptBuilder
from db.schema_fetcher import ColumnInfo, DatabaseSchema, TableSchema
from evaluation.ragas_eval import context_table_recall, execution_results_match
from llm.base import Message
from llm.gemini_llm import GeminiLLM


def column(name: str, data_type: str = "varchar", key: str = "") -> ColumnInfo:
    return ColumnInfo(
        name=name,
        data_type=data_type,
        nullable=True,
        key=key,
        default=None,
        extra="",
    )


class PromptBuilderTests(unittest.TestCase):
    def setUp(self):
        customers = TableSchema(
            name="customers",
            columns=[column("customer_id", "int", "PRI"), column("customer_name")],
        )
        orders = TableSchema(
            name="orders",
            columns=[
                column("order_id", "int", "PRI"),
                column("customer_id", "int"),
                column("total_amount", "decimal"),
            ],
            foreign_keys=[
                {
                    "column": "customer_id",
                    "ref_table": "customers",
                    "ref_column": "customer_id",
                }
            ],
        )
        audit = TableSchema(
            name="audit_log",
            columns=[column(f"metadata_field_{index}") for index in range(100)],
        )
        self.schema = DatabaseSchema(
            database="shop",
            tables={"audit_log": audit, "customers": customers, "orders": orders},
        )

    def test_selects_relevant_table_and_fk_neighbour(self):
        prompt = PromptBuilder(self.schema, schema_token_budget=500).build_with_context(
            "Top customers by total order amount"
        )
        self.assertIn("orders", prompt.selected_tables)
        self.assertIn("customers", prompt.selected_tables)
        self.assertNotIn("audit_log", prompt.selected_tables)
        self.assertGreater(prompt.estimated_input_tokens, 0)

    def test_question_is_delimited(self):
        prompt = PromptBuilder(self.schema).build_with_context("Count all orders")
        self.assertIn("<question>\nCount all orders\n</question>", prompt.messages[1].content)


class EvaluationMetricTests(unittest.TestCase):
    def test_execution_match_ignores_row_order_and_aliases(self):
        self.assertTrue(
            execution_results_match(
                ["customer", "amount"],
                [("A", 2.0), ("B", 3.0)],
                ["name", "total"],
                [("B", 3), ("A", 2)],
            )
        )

    def test_context_table_recall(self):
        recall = context_table_recall(
            "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.customer_id",
            ["orders"],
            ["orders", "customers", "products"],
        )
        self.assertEqual(recall, 0.5)


class GeminiAdapterTests(unittest.TestCase):
    def test_request_uses_system_instruction_and_model_role(self):
        llm = GeminiLLM(api_key="test-key")
        try:
            contents, config = llm._prepare_request(
                [
                    Message(role="system", content="Return SQL only."),
                    Message(role="user", content="Count orders"),
                    Message(role="assistant", content="SELECT COUNT(*) FROM orders"),
                ]
            )
            self.assertEqual(llm.model, "gemini-3.5-flash")
            self.assertEqual(config.system_instruction, "Return SQL only.")
            self.assertEqual([content.role for content in contents], ["user", "model"])
            self.assertEqual(config.max_output_tokens, 1024)
        finally:
            llm.close()


if __name__ == "__main__":
    unittest.main()
