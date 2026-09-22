import io
import unittest

from app import extract_text


class ExtractTextTests(unittest.TestCase):
    def test_extracts_csv_text(self):
        uploaded = io.BytesIO(b"name,role\nAlice,Engineer\n")
        uploaded.name = "people.csv"

        text = extract_text(uploaded)

        self.assertIn("name", text)
        self.assertIn("Alice", text)
        self.assertIn("Engineer", text)

    def test_extracts_rtf_text(self):
        uploaded = io.BytesIO(rb"{\rtf1\ansi Hello \b world\b0.}")
        uploaded.name = "note.rtf"

        text = extract_text(uploaded)

        self.assertIn("Hello", text)
        self.assertIn("world", text)


if __name__ == "__main__":
    unittest.main()
