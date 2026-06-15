"""Dependency-free JSON-schema validation for tool-call arguments.

A deliberately small validator over the JSON-schema subset that tool argument
specs actually use: ``type``, ``properties``, ``required``, ``enum``, ``items``,
and nested objects/arrays. It returns human-readable error strings (consumed by
the structured-tool-use penalty and surfaced to the agent), not exceptions.

No third-party dependency: ``jsonschema`` is not guaranteed in the runtime image,
and the tool schemas here are simple enough that a focused validator is both
sufficient and easier to reason about than pulling a draft-aware library.
"""

from __future__ import annotations

from typing import Any

# JSON-schema "type" -> Python type(s). "integer" excludes bool (bool is an int
# subclass in Python, but JSON booleans are not integers); "number" likewise.
_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "boolean": (bool,),
    "null": (type(None),),
    "integer": (int,),
    "number": (int, float),
}


class SchemaError(ValueError):
    """Raised for a malformed schema (a build-time error, not a data error)."""


def _type_matches(value: Any, json_type: str) -> bool:
    expected = _JSON_TYPES.get(json_type)
    if expected is None:
        raise SchemaError(f"unsupported JSON-schema type: {json_type!r}")
    if json_type in ("integer", "number"):
        # Reject bool, which is a subclass of int but a distinct JSON type.
        if isinstance(value, bool):
            return False
        return isinstance(value, expected)
    if json_type == "boolean":
        return isinstance(value, bool)
    return isinstance(value, expected)


def validate_against_schema(
    value: Any, schema: dict, *, path: str = "$"
) -> list[str]:
    """Validate ``value`` against a JSON-schema subset; return error strings.

    An empty list means valid. ``path`` is the JSON path of ``value`` used in
    error messages (callers pass ``"$"`` for the root).

    Supported keywords: ``type`` (single string), ``enum``, ``properties``,
    ``required``, ``items``. Unknown keywords are ignored (permissive), but an
    unknown ``type`` value raises :class:`SchemaError`.
    """
    errors: list[str] = []

    json_type = schema.get("type")
    if json_type is not None:
        if not isinstance(json_type, str):
            raise SchemaError(f"{path}: 'type' must be a string, got {json_type!r}")
        if not _type_matches(value, json_type):
            errors.append(f"{path}: expected {json_type}, got {_pytype_name(value)}")
            # Type is wrong; deeper structural checks would be noise.
            return errors

    if "enum" in schema:
        allowed = schema["enum"]
        if not isinstance(allowed, list):
            raise SchemaError(f"{path}: 'enum' must be a list")
        if value not in allowed:
            errors.append(f"{path}: {value!r} is not one of {allowed!r}")

    if json_type == "object" or (json_type is None and isinstance(value, dict)):
        errors.extend(_validate_object(value, schema, path))

    if json_type == "array" or (json_type is None and isinstance(value, list)):
        errors.extend(_validate_array(value, schema, path))

    return errors


def _validate_object(value: Any, schema: dict, path: str) -> list[str]:
    if not isinstance(value, dict):
        return []  # type mismatch already reported (or no type constraint)
    errors: list[str] = []

    required = schema.get("required", [])
    if not isinstance(required, list):
        raise SchemaError(f"{path}: 'required' must be a list")
    for key in required:
        if key not in value:
            errors.append(f"{path}: missing required property {key!r}")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise SchemaError(f"{path}: 'properties' must be an object")
    for key, subschema in properties.items():
        if key in value:
            child = f"{path}.{key}"
            errors.extend(validate_against_schema(value[key], subschema, path=child))

    return errors


def _validate_array(value: Any, schema: dict, path: str) -> list[str]:
    if not isinstance(value, list):
        return []
    items_schema = schema.get("items")
    if items_schema is None:
        return []
    if not isinstance(items_schema, dict):
        raise SchemaError(f"{path}: 'items' must be a schema object")
    errors: list[str] = []
    for i, item in enumerate(value):
        errors.extend(validate_against_schema(item, items_schema, path=f"{path}[{i}]"))
    return errors


def validate_arguments(arguments: Any, parameters_schema: dict) -> list[str]:
    """Validate a tool call's ``arguments`` against its ``parameters`` schema.

    Convenience wrapper over :func:`validate_against_schema` that first asserts
    the arguments are a JSON object (the only valid top level for an
    OpenAI-style function ``parameters`` schema).
    """
    if not isinstance(arguments, dict):
        return [f"$: expected object arguments, got {_pytype_name(arguments)}"]
    return validate_against_schema(arguments, parameters_schema, path="$")


def _pytype_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__
