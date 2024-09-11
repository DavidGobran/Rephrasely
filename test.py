import unittest
from unittest.mock import patch, MagicMock
from app import main

class TestApp(unittest.TestCase):

    @patch('cs_553_case_study_1.app.st')
    @patch('cs_553_case_study_1.app.respond')
    def test_main_with_local_model(self, mock_respond, mock_st):
        # Mock Streamlit functions
        mock_st.markdown = MagicMock()
        mock_st.text_area = MagicMock(side_effect=["You are a friendly Chatbot who paraphrases text.", "Test input text"])
        mock_st.checkbox = MagicMock(return_value=True)
        mock_st.slider = MagicMock(side_effect=[512, 0.7, 0.95])
        mock_st.button = MagicMock(side_effect=[True, False])
        mock_st.spinner = MagicMock()
        mock_st.success = MagicMock()
        mock_st.error = MagicMock()

        # Mock the respond function
        mock_respond.return_value = "Paraphrased text"

        # Run the main function
        main()

        # Assertions to check if the functions were called correctly
        mock_st.markdown.assert_called_once()
        mock_st.text_area.assert_any_call("System message", "You are a friendly Chatbot who paraphrases text.")
        mock_st.checkbox.assert_called_once_with("Use local model", value=False)
        mock_st.slider.assert_any_call('Max new tokens', 1, 2048, 512)
        mock_st.slider.assert_any_call('Temperature', 0.1, 1.0, 0.1)
        mock_st.slider.assert_any_call('Top-p (nucleus sampling)', 0.1, 1.0, 0.95)
        mock_st.text_area.assert_any_call("Enter the text to paraphrase:", "", height=150)
        mock_st.button.assert_any_call("Submit")
        mock_st.spinner.assert_called_once_with("Processing...")
        mock_respond.assert_called_once_with("Test input text", system_message="You are a friendly Chatbot who paraphrases text.", max_tokens=512, temperature=0.7, top_p=0.95, use_local_model=True)
        mock_st.success.assert_called_once_with("Text successfully paraphrased!")

if __name__ == '__main__':
    unittest.main()