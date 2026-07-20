import pytest

from src import Spiderbot


class TestSpiderBot:
    def test_spiderbot_should_construct(self):
        spiderbot = Spiderbot()
        assert spiderbot is not None