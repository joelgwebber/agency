import unittest
from typing import List

from research.src import foo

print(foo.bar())


def sort_numbers(numbers: List[int]) -> List[int]:
    """Sort a list of numbers in ascending order."""
    return sorted(numbers)


def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


class TestDemoFunctions(unittest.TestCase):

    def test_sort_numbers(self):
        """Test number sorting."""
        input_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        expected = [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
        self.assertEqual(sort_numbers(input_list), expected)
        self.assertEqual(sort_numbers([]), [])  # Empty list case

    def test_reverse_string(self):
        """Test string reversal."""
        self.assertEqual(reverse_string("hello"), "olleh")
        self.assertEqual(reverse_string(""), "")  # Empty string case
        self.assertEqual(reverse_string("a"), "a")  # Single char case


if __name__ == "__main__":
    unittest.main()
