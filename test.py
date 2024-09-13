import unittest
from unittest.mock import patch
from inference import respond

class TestRespondFunction(unittest.TestCase):
    # GitHub Copilot generated this test case
    @patch('app.pipe')
    def test_respond_with_local_model(self, mock_pipe):
        # Mock the pipe function
        mock_pipe.return_value = [{"generated_text": [{"content": "Inference cancelled."}]}]

        # Set stop_inference to True
        global stop_inference
        stop_inference = True

        # Call the respond function with use_local_model=True
        response = respond(
            message="Test input",
            use_local_model=True
        )

        # Assert that the pipe function was called
        mock_pipe.assert_called_once()

        # Assert the response is "Inference cancelled."
        self.assertEqual(response, "Inference cancelled.")

if __name__ == '__main__':
    unittest.main()