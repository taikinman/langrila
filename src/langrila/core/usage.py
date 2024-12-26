from typing import ItemsView, Iterator, KeysView, ValuesView

from .pydantic import BaseModel


class Usage(BaseModel):
    model_name: str | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        if self.model_name and other.model_name and self.model_name != other.model_name:
            raise ValueError(
                f"Cannot add Usage objects with different model names: {self.model_name} and {other.model_name}"
            )

        return Usage(
            model_name=self.model_name,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def __radd__(self, other: "Usage") -> "Usage":
        return self.__add__(other)

    def __sub__(self, other: "Usage") -> "Usage":
        if self.model_name and other.model_name and self.model_name != other.model_name:
            raise ValueError(
                f"Cannot subtract Usage objects with different model names: {self.model_name} and {other.model_name}"
            )

        return Usage(
            model_name=self.model_name,
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
        )

    def __rsub__(self, other: "Usage") -> "Usage":
        return self.__sub__(other)


class NamedUsage:
    def __init__(self) -> None:
        """
        Initialize the TypedDict.
        """
        self._data: dict[str, Usage] = {}

    def __setitem__(self, key: str, value: Usage) -> None:
        """
        Set a key-value pair in the dictionary, ensuring the value matches the generic type.

        Args:
            key (str): The key for the dictionary.
            value (Usage): The value to associate with the key.

        Raises:
            TypeError: If the value is not of the expected type.
        """
        if not isinstance(value, Usage):
            raise TypeError(f"Value must be of type Usage, not {type(value).__name__}")

        self._data[key] = value

    def __getitem__(self, key: str) -> Usage:
        """
        Retrieve the value associated with the given key.

        Args:
            key (str): The key to look up.

        Returns:
            Usage: The value associated with the key.

        Raises:
            KeyError: If the key is not found in the dictionary.
        """
        return self._data[key]

    def __delitem__(self, key: str) -> None:
        """
        Delete a key-value pair from the dictionary.

        Args:
            key (str): The key to delete.

        Raises:
            KeyError: If the key is not found in the dictionary.
        """
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the dictionary.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over the dictionary keys.

        Returns:
            Iterator: An iterator over the keys of the dictionary.
        """
        return iter(self._data)

    def __len__(self) -> int:
        """
        Return the number of items in the dictionary.

        Returns:
            int: The number of key-value pairs in the dictionary.
        """
        return len(self._data)

    def __add__(self, other: "NamedUsage") -> "NamedUsage":
        result = NamedUsage()
        all_keys = set(self.keys()) | set(other.keys())
        for key in all_keys:
            if key in self and key in other:
                result[key] = self[key] + other[key]
            elif key in self:
                result[key] = self[key]
            else:
                result[key] = other[key]

        return result

    def __sub__(self, other: "NamedUsage") -> "NamedUsage":
        result = NamedUsage()
        all_keys = set(self.keys()) | set(other.keys())
        for key in all_keys:
            if key in self and key in other:
                result[key] = self[key] - other[key]
            elif key in self:
                result[key] = self[key]
            else:
                result[key] = other[key]

        return result

    def __radd__(self, other: "NamedUsage") -> "NamedUsage":
        return self.__add__(other)

    def __rsub__(self, other: "NamedUsage") -> "NamedUsage":
        return self.__sub__(other)

    def keys(self) -> KeysView[str]:
        """Return the keys of the dictionary."""
        return self._data.keys()

    def values(self) -> ValuesView[Usage]:
        """Return the values of the dictionary."""
        return self._data.values()

    def items(self) -> ItemsView[str, Usage]:
        """Return the items (key-value pairs) of the dictionary."""
        return self._data.items()
