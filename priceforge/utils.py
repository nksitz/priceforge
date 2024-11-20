from enum import Enum
from typing import Type, TypeVar, Union


E = TypeVar("E", bound=Enum)


def parse_enum(
    value: Union[str, E], enum_class: Type[E], case_sensitive: bool = False
) -> E:
    if isinstance(value, enum_class):
        return value

    if isinstance(value, str):
        # Get all possible enum names
        available = [e.name for e in enum_class]

        try:
            if case_sensitive:
                return enum_class[value]
            else:
                # Case-insensitive match
                lookup = {k.upper(): v for k, v in enum_class.__members__.items()}
                return lookup[value.upper()]
        except KeyError:
            # Format available options with original case
            options = ", ".join(available)
            raise ValueError(
                f"Unknown value '{value}' for {enum_class.__name__}. "
                f"Available options: {options}"
            )

    raise TypeError(
        f"Value must be string or {enum_class.__name__}, " f"got {type(value).__name__}"
    )
