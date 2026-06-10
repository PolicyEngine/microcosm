"""Entity schemas, link declarations, and variable metadata for the kernel.

An :class:`EntitySchema` declares the entity structure of a frame once, at
assembly: a person entity plus the group entities persons belong to. Linkage
columns follow a fixed convention — the person table carries one membership
column ``person_{group}_id`` per group, and each group table carries its own
id column ``{group}_id`` — so every operator reads structure the same way.

:class:`LinkSpec` declares an *association* between two entities (e.g. a
``jobs`` link between persons and firms): a many-to-many relation carried as
its own table, distinct from the partition semantics of group membership.
Links are a documented placeholder today — the schema declares them and the
frame validates their tables, but link-aware operators come later.

:class:`VariableMetadata` records what a variable is (owning entity, dtype
kind, period semantics) so tools resolve it through metadata instead of
guessing.
"""

from dataclasses import dataclass

__all__ = ["EntitySchema", "LinkSpec", "VariableMetadata"]

_DTYPE_KINDS: tuple[str, ...] = ("float", "int", "bool", "str")
_PERIOD_SEMANTICS: tuple[str, ...] = ("year", "month", "point")


@dataclass(frozen=True)
class LinkSpec:
    """Declaration of an association (link) between two entities.

    A link is a many-to-many relation carried as its own table — e.g. a
    ``jobs`` link between persons and firms, where one person may hold many
    jobs and one firm employs many persons. The link table is keyed by the
    link's name in a frame's ``tables`` and must carry the two linked
    entities' id columns (``{left}_id`` / ``{right}_id`` per the schema's
    id-column convention), each validated against the linked table's ids.

    This is a documented placeholder: the schema declares links and the
    frame validates their tables on construction, but link-aware operators
    (broadcast through links, link-preserving ``select``/``concat``,
    link targets outside the partition entities) come later.

    Attributes:
        name: Name of the link (and of its table in a frame's ``tables``).
        left_entity: Entity on the left side of the relation.
        right_entity: Entity on the right side of the relation.

    Raises:
        ValueError: If any field is empty.
    """

    name: str
    left_entity: str
    right_entity: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("LinkSpec.name must be non-empty.")
        if not self.left_entity:
            raise ValueError("LinkSpec.left_entity must be non-empty.")
        if not self.right_entity:
            raise ValueError("LinkSpec.right_entity must be non-empty.")


@dataclass(frozen=True, kw_only=True)
class EntitySchema:
    """Entity structure of a frame: one person entity plus group entities.

    Attributes:
        person_entity: Name of the person-level entity. Defaults to
            ``"person"``.
        group_entities: Names of the group entities persons belong to (e.g.
            ``("household", "tax_unit")``). Order is preserved.
        links: Declared associations between entities (see
            :class:`LinkSpec`). A documented placeholder: declared links are
            validated here and their tables are validated by the frame, but
            link-aware operators come later.

    Raises:
        ValueError: If ``group_entities`` is empty, contains duplicates or
            empty names, or contains the person entity; or if ``links``
            contains duplicate names, names that collide with an entity, or
            sides that are not declared entities.
    """

    person_entity: str = "person"
    group_entities: tuple[str, ...]
    links: tuple[LinkSpec, ...] = ()

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
        self._validate_links()

    def _validate_links(self) -> None:
        names = [link.name for link in self.links]
        if len(set(names)) != len(names):
            duplicated = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"Link names must be unique; duplicated: {duplicated}.")
        entities = set(self.entities)
        for link in self.links:
            if link.name in entities:
                raise ValueError(
                    f"Link name {link.name!r} collides with an entity name."
                )
            for side in (link.left_entity, link.right_entity):
                if side not in entities:
                    raise ValueError(
                        f"Link {link.name!r} references unknown entity "
                        f"{side!r}; declared entities: {list(self.entities)}."
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

    def entity_id_column(self, entity: str) -> str:
        """Id column for any declared entity: ``{entity}_id``.

        Args:
            entity: The person entity or a declared group entity.

        Returns:
            :attr:`person_id_column` for the person entity, otherwise
            :meth:`id_column`.

        Raises:
            ValueError: If ``entity`` is not declared by the schema.
        """
        if entity == self.person_entity:
            return self.person_id_column
        return self.id_column(entity)

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
