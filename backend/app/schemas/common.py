"""Shared base class for API request/response schemas."""
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """camelCase over the wire, snake_case in Python.

    `from_attributes` lets these be built directly from internal
    `app.models` instances (e.g. `TestCaseOut.model_validate(test_case)`),
    keeping the API contract separate from internal representation.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )
