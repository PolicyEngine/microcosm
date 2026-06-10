"""Entity schemas and variable metadata for the kernel.

An :class:`EntitySchema` declares the entity structure of a bundle once, at
assembly: a person entity plus the group entities persons belong to. Linkage
columns follow a fixed convention — the person table carries one membership
column ``person_{group}_id`` per group, and each group table carries its own
id column ``{group}_id`` — so every operator reads structure the same way.

:class:`VariableMetadata` records what a variable is (owning entity, dtype
kind, period semantics) so tools resolve it through metadata instead of
guessing.
"""

from dataclasses import dataclass

__all__ = ["EntitySchema", "VariableMetadata"]

_DTYPE_KINDS: tuple[str, ...] = ("float", "int", "bool", "str")
_PERIOD_SEMANTICS: tuple[str, ...] = ("year", "month", "point")


@dataclass(frozen=True, kw_only=True)
class EntitySchema:
    """Entity structure of a bundle: one person entity plus group entities.

    Attributes:
        person_entity: Name of the person-level entity. Defaults to
            ``"person"``.
        group_entities: Names of the group entities persons belong to (e.g.
            ``("household", "tax_unit")``). Order is preserved.

    Raises:
        ValueError: If ``group_entities`` is empty, contains duplicates or
            empty names, or contains the person entity.
    """

    person_entity: str = "person"
    group_entities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.person_entity:
            raise ValueError("person_entity must be a non-empty name.")
        if not self.group_entities:
            raise ValueError("group_entities must declare at least one group.")
        if any(not name for name in self.group_entities):
            raise ValueError("group_entities must not contain empty names.")
        if len(set(self.group_entities)) != len(self.group_entities):
            raise ValueError(
                f"group_entities must be unique, got {self.group_entities!r}."
            )
        if self.person_entity in self.group_entities:
            raise ValueError(
                f"person_entity {self.person_entity!r} cannot also be a group entity."
            )

    @property
    def entities(self) -> tuple[str, ...]:
        """All entity names: the person entity followed by the group entities."""
        return (self.person_entity, *self.group_entities)

    @property
    def person_id_column(self) -> str:
        """Id column on the person table: ``{person_entity}_id``."""
        return f"{self.person_entity}_id"

    def membership_column(self, group: str) -> str:
        """Membership column on the person table for ``group``.

        Args:
            group: A declared group entity.

        Returns:
            The column name ``{person_entity}_{group}_id`` (with the default
            person entity: ``person_{group}_id``).

        Raises:
            ValueError: If ``group`` is not a declared group entity.
        """
        self._require_group(group)
        return f"{self.person_entity}_{group}_id"

    def id_column(self, group: str) -> str:
        """Id column on the ``group`` table: ``{group}_id``.

        Args:
            group: A declared group entity.

        Returns:
            The column name ``{group}_id``.

        Raises:
            ValueError: If ``group`` is not a declared group entity.
        """
        self._require_group(group)
        return f"{group}_id"

    def _require_group(self, group: str) -> None:
        if group not in self.group_entities:
            raise ValueError(
                f"Unknown group entity {group!r}; declared groups: "
                f"{list(self.group_entities)}."
            )


@dataclass(frozen=True)
class VariableMetadata:
    """What a variable is: owning entity, dtype kind, and period semantics.

    Attributes:
        name: Variable name.
        entity: Entity the variable lives on (one row per entity record).
        dtype: Dtype kind, one of ``"float"``, ``"int"``, ``"bool"``,
            ``"str"``.
        period: Period semantics, one of ``"year"`` (annual flow),
            ``"month"`` (monthly flow), ``"point"`` (point-in-time state).

    Raises:
        ValueError: If ``name`` or ``entity`` is empty, or ``dtype`` /
            ``period`` is not one of the allowed kinds.
    """

    name: str
    entity: str
    dtype: str
    period: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("VariableMetadata.name must be non-empty.")
        if not self.entity:
            raise ValueError("VariableMetadata.entity must be non-empty.")
        if self.dtype not in _DTYPE_KINDS:
            raise ValueError(
                f"VariableMetadata.dtype must be one of {_DTYPE_KINDS}, "
                f"got {self.dtype!r}."
            )
        if self.period not in _PERIOD_SEMANTICS:
            raise ValueError(
                f"VariableMetadata.period must be one of {_PERIOD_SEMANTICS}, "
                f"got {self.period!r}."
            )
