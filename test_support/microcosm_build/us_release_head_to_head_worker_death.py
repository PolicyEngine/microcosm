"""Small importable seams for an abruptly terminated scorer worker."""

import os


class OneHouseholdFrame:
    def n(self, entity: str) -> int:
        return 1


def initialize_slice_worker(*args) -> None:
    pass


def exit_slice_worker(task: tuple[int, int]) -> None:
    os._exit(97)
