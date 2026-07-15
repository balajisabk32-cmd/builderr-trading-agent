import unittest

from market_open import market_is_open


class MarketOpenTest(unittest.TestCase):
    def test_open_status(self):
        self.assertTrue(market_is_open({"status": "open"}))
        self.assertTrue(market_is_open({"status": "OPEN"}))

    def test_closed_or_missing_status(self):
        self.assertFalse(market_is_open({"status": "closed"}))
        self.assertFalse(market_is_open({}))
        self.assertFalse(market_is_open(None))


if __name__ == "__main__":
    unittest.main()
