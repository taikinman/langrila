import re
from typing import Any

from ...base import BaseMetadataFilter


class SQLiteMetadataFilter(BaseMetadataFilter):
    """
    Apply a WHERE clause in SQLite format to a metadata dictionary.
    """

    def __init__(self, where: str):
        where = re.sub("%", "", where)
        self.conditions: list[tuple[Any, str, Any]] = self._parse_where_clause(where_clause=where)

    def _parse_value(self, value):
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]  # Remove the single quotes
        elif value.startswith('"') and value.endswith('"'):
            value = value[1:-1]  # Remove the double quotes
        return value

    def _parse_in_clause(self, field, values):
        values = values.strip()[1:-1]  # Remove the surrounding parentheses
        value_list = [self._parse_value(v.strip()) for v in values.split(",")]
        return [field, "IN", value_list]

    def _parse_not_in_clause(self, field, values):
        values = values.strip()[1:-1]  # Remove the surrounding parentheses
        value_list = [self._parse_value(v.strip()) for v in values.split(",")]
        return [field, "NOT IN", value_list]

    def _parse_where_clause(self, where_clause: str) -> list[tuple[Any, str, Any]]:
        conditions = []
        stack = []

        # Separate IN clause to distinguish brackets in IN clause from other brackets
        not_in_clauses = re.findall(r"\w+\s+NOT IN\s+\([^)]*\)", where_clause, flags=re.IGNORECASE)
        for not_in_clause in not_in_clauses:
            where_clause = where_clause.replace(not_in_clause, f"NOT_IN_CLAUSE_{len(stack)}")
            stack.append(not_in_clause)

        in_clauses = re.findall(r"\w+\s+IN\s+\([^)]*\)", where_clause, flags=re.IGNORECASE)
        for in_clause in in_clauses:
            where_clause = where_clause.replace(in_clause, f"IN_CLAUSE_{len(stack)}")
            stack.append(in_clause)

        tokens = re.split(r"(\(|\)|\s+AND\s+|\s+OR\s+)", where_clause, flags=re.IGNORECASE)
        tokens = [token.strip() for token in tokens if token.strip()]

        for token in tokens:
            if token == "(":
                stack.append(conditions)
                conditions = []
            elif token == ")":
                last_conditions = conditions
                conditions = stack.pop()
                conditions.append(last_conditions)
            elif token.upper() in ["AND", "OR"]:
                conditions.append(token.upper())
            elif token.startswith("IN_CLAUSE_"):
                index = int(token.split("_")[-1])
                in_clause = stack[index]
                in_match = re.match(r"(\w+)\s+IN\s+(\(.+\))", in_clause, re.IGNORECASE)
                if in_match:
                    field, values = in_match.groups()
                    conditions.append(self._parse_in_clause(field, values))
            elif token.startswith("NOT_IN_CLAUSE_"):
                index = int(token.split("_")[-1])
                not_in_clause = stack[index]
                not_in_match = re.match(r"(\w+)\s+NOT IN\s+(\(.+\))", not_in_clause, re.IGNORECASE)
                if not_in_match:
                    field, values = not_in_match.groups()
                    conditions.append(self._parse_not_in_clause(field, values))
            else:
                match = re.match(r"(\w+)\s*(>=|<=|!=|<>|=|>|<|LIKE)\s*(.+)", token, re.IGNORECASE)
                if match:
                    field, operator, value = match.groups()
                    value = self._parse_value(value)
                    conditions.append([field, operator.upper(), value])

        return conditions

    def _apply_conditions(
        self, item: dict[str, Any], conditions: list[tuple[Any, str, Any]]
    ) -> bool:
        stack = []
        for condition in conditions:
            if condition in ["AND", "OR"]:
                stack.append(condition)
            else:
                if isinstance(condition, list) and ("AND" in condition or "OR" in condition):
                    stack.append(self._apply_conditions(item, condition))
                else:
                    field: str
                    operator: str
                    value: str
                    field, operator, value = condition
                    item_value = item.get(field)

                    if item_value is None:
                        result = False
                    else:
                        if operator == "=":
                            result = item_value == value
                        elif operator == "!=" or operator == "<>":
                            result = item_value != value
                        elif operator == ">":
                            if value.isdigit():
                                result = float(item_value) > float(value)
                            else:
                                result = item_value > value
                        elif operator == "<":
                            if value.isdigit():
                                result = float(item_value) < float(value)
                            else:
                                result = item_value < value
                        elif operator == ">=":
                            if value.isdigit():
                                result = float(item_value) >= float(value)
                            else:
                                result = item_value >= value
                        elif operator == "<=":
                            if value.isdigit():
                                result = float(item_value) <= float(value)
                            else:
                                result = item_value <= value
                        elif operator == "LIKE":
                            pattern_split = re.compile(r",|、|\s|　")
                            values = pattern_split.split(value)
                            result = any([bool(re.search(v, item_value)) for v in values if v])
                        elif operator == "NOT IN":
                            pattern_split = re.compile(r",|、|\s|　")
                            item_values = pattern_split.split(item_value)
                            value = set(value)
                            result = all([v not in value for v in item_values])
                        elif operator == "IN":
                            pattern_split = re.compile(r",|、|\s|　")
                            item_values = pattern_split.split(item_value)
                            value = set(value)
                            result = any([v in value for v in item_values])

                    stack.append(result)

        # Evaluate the condition stack
        while len(stack) > 1:
            left = stack.pop(0)
            operator = stack.pop(0)
            right = stack.pop(0)
            if operator == "AND":
                stack.insert(0, left and right)
            elif operator == "OR":
                stack.insert(0, left or right)

        if stack:
            return stack[0]
        else:
            return False

    def run(self, metadata: dict[str, Any]) -> bool:
        return self._apply_conditions(metadata, self.conditions)
