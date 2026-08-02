"""Tests the core functionality of the Spiderbot."""

from src import Spiderbot


class TestSpiderBot:
    """Tests the core functionality of the Spiderbot."""

    def test_spiderbot_should_construct(self):
        """Test that a Spiderbot can be constructed successfully."""
        spiderbot = Spiderbot()
        assert spiderbot is not None
