from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


INSTANCE_SCHEMA_VERSION = "bilgeval.instance/v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(StrictModel):
    source: str = Field(min_length=1)
    source_revision: str | None = None
    source_record_id: str | None = None
    source_record_sha256: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    converter: str | None = None


class ChoiceRecord(StrictModel):
    label: str = Field(min_length=1)
    text: str = Field(min_length=1)


class MultipleChoiceInput(StrictModel):
    question: str = Field(min_length=1)
    choices: list[ChoiceRecord] = Field(min_length=2)


class MultipleChoiceTarget(StrictModel):
    answer_label: str = Field(min_length=1)
    answer_text: str | None = None


class FreeResponseInput(StrictModel):
    question: str = Field(min_length=1)


class FreeResponseTarget(StrictModel):
    references: list[str] = Field(min_length=1)


class CodeGenerationInput(StrictModel):
    prompt: str = Field(min_length=1)


class CodeGenerationTarget(StrictModel):
    test: str = Field(min_length=1)
    entry_point: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    canonical_solution: str | None = None


class MessageRecord(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class FunctionDefinition(StrictModel):
    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any]

    @model_validator(mode="after")
    def validate_parameters(self):
        if self.parameters.get("type") != "object":
            raise ValueError("Function parameters must use JSON Schema type 'object'.")
        properties = self.parameters.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("Function parameters must define a properties object.")
        required = self.parameters.get("required", [])
        if not isinstance(required, list) or any(
            not isinstance(name, str) for name in required
        ):
            raise ValueError("Function required parameters must be a list of names.")
        unknown_required = set(required) - set(properties)
        if unknown_required:
            raise ValueError(
                "Function required parameters are missing from properties: "
                + ", ".join(sorted(unknown_required))
            )
        return self


class FunctionToolRecord(StrictModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCallExpectation(StrictModel):
    name: str = Field(min_length=1)
    arguments: dict[str, list[Any]]

    @model_validator(mode="after")
    def validate_accepted_values(self):
        empty = [name for name, values in self.arguments.items() if not values]
        if empty:
            raise ValueError(
                "Function-call expected arguments must define accepted values: "
                + ", ".join(sorted(empty))
            )
        return self


class FunctionCallInput(StrictModel):
    messages: list[MessageRecord] = Field(min_length=1)
    tools: list[FunctionToolRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tool_names(self):
        names = [tool.function.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("Function tool names must be unique within an instance.")
        return self


class FunctionCallTarget(StrictModel):
    calls: list[FunctionCallExpectation] = Field(default_factory=list)
    should_call: bool
    call_order: Literal["ordered", "unordered"] = "unordered"
    scoring_mode: Literal["exact", "call_presence"] = "exact"

    @model_validator(mode="after")
    def validate_call_intent(self):
        if self.scoring_mode == "exact" and self.should_call != bool(self.calls):
            raise ValueError("should_call must agree with whether target calls are present.")
        if self.scoring_mode == "call_presence" and self.calls:
            raise ValueError("call_presence targets must not define exact calls.")
        return self


class CanonicalInstanceBase(StrictModel):
    schema_version: Literal[INSTANCE_SCHEMA_VERSION]
    id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    language: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance


class MultipleChoiceInstance(CanonicalInstanceBase):
    task_type: Literal["multiple_choice"]
    input: MultipleChoiceInput
    target: MultipleChoiceTarget

    @model_validator(mode="after")
    def validate_target(self):
        labels = [choice.label for choice in self.input.choices]
        if len(labels) != len(set(labels)):
            raise ValueError("Multiple-choice labels must be unique.")
        if self.target.answer_label not in labels:
            raise ValueError(
                f"answer_label {self.target.answer_label!r} is not present in the choices."
            )
        expected_text = next(
            choice.text
            for choice in self.input.choices
            if choice.label == self.target.answer_label
        )
        if self.target.answer_text is not None and self.target.answer_text != expected_text:
            raise ValueError("answer_text must match the choice selected by answer_label.")
        return self


class FreeResponseInstance(CanonicalInstanceBase):
    task_type: Literal["free_response"]
    input: FreeResponseInput
    target: FreeResponseTarget

    @model_validator(mode="after")
    def validate_references(self):
        if any(not reference.strip() for reference in self.target.references):
            raise ValueError("Free-response references must not be empty.")
        return self


class CodeGenerationInstance(CanonicalInstanceBase):
    task_type: Literal["code_generation"]
    input: CodeGenerationInput
    target: CodeGenerationTarget


class FunctionCallInstance(CanonicalInstanceBase):
    task_type: Literal["function_call"]
    input: FunctionCallInput
    target: FunctionCallTarget

    @model_validator(mode="after")
    def validate_expected_functions(self):
        tool_names = {tool.function.name for tool in self.input.tools}
        unknown = {call.name for call in self.target.calls} - tool_names
        if unknown:
            raise ValueError(
                "Expected function calls reference unknown tools: "
                + ", ".join(sorted(unknown))
            )
        return self


CanonicalInstance = Annotated[
    Union[
        MultipleChoiceInstance,
        FreeResponseInstance,
        CodeGenerationInstance,
        FunctionCallInstance,
    ],
    Field(discriminator="task_type"),
]
CANONICAL_INSTANCE_ADAPTER = TypeAdapter(CanonicalInstance)


def parse_canonical_instance(value: dict[str, Any]) -> CanonicalInstance:
    return CANONICAL_INSTANCE_ADAPTER.validate_python(value)
