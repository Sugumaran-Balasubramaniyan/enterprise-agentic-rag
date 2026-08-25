import unittest

class TestApiModule(unittest.TestCase):
    def test_schemas_definition(self):
        from app.api.schemas import QueryRequest, QueryResponse
        req = QueryRequest(query="What is HNSW?")
        self.assertEqual(req.query, "What is HNSW?")
        self.assertEqual(req.user_role, "standard_user")

if __name__ == "__main__":
    unittest.main()
