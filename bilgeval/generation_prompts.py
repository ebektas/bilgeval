from typing import Any

from .reader import Example


BUILTIN_GENERATION_PROMPTS = {
    "none": "{question}{choices_block}",
    "turkish_mc_v1": """Aşağıdaki soruyu doğru seçenekle cevapla.

Soru: {question}
{choices_block}
Cevap:""",
    "turkish_mc_cot_v1": """Aşağıdaki çoktan seçmeli soruyu adım adım çöz. Son satırı `Cevap: X` biçiminde yaz; X doğru seçeneğin harfi olmalıdır.

Soru: {question}
{choices_block}
Cevap:""",
    "turkish_mc_cot_v2": """Aşağıdaki çoktan seçmeli soruyu adım adım çöz. Son satırı `Cevap: X` biçiminde yaz; X doğru seçeneğin harfi olmalıdır.

Soru: {question}
{choices_block}
Cevap: Adım adım düşünelim.""",
    "turkish_continuation_v1": "{question}",
    "english_mc_v1": """Answer the following multiple-choice question with the correct option.

Question: {question}
{choices_block}
Answer:""",
    "turkish_integer_answer_v1": """Aşağıdaki matematik problemini çöz. Yalnızca nihai tam sayı cevabı yaz.

Problem: {question}
Cevap:""",
    "turkish_math_cot_v1": """Aşağıdaki matematik problemini adım adım çöz. Son satırı `Cevap: ...` biçiminde yaz ve yalnızca nihai matematiksel ifadeyi bu satırda ver.

Problem: {question}
Cevap:""",
    "turkish_mc_cot_strict_v1": """Aşağıdaki çoktan seçmeli soruyu adım adım çöz. Yanıtının son satırı `Cevap: X` biçiminde olmalı; X doğru seçeneğin harfidir.

Soru: {question}
{choices_block}""",
    "turkish_math_cot_strict_v1": """Aşağıdaki matematik problemini adım adım çöz. Yanıtının son satırı `Cevap: ...` biçiminde olmalı; bu satırda yalnızca nihai cevabı ver.

Problem: {question}""",
    "turkish_gsm8k_cot_v1": """Aşağıdaki matematik problemini adım adım çöz. Son cümlede nihai sayıyı `Cevap N'dir.` biçiminde yaz.

Problem: {question}
Cevap:""",
    "aime25_reasoning_en_v1": """{question}
Please reason step by step, and put your final answer within \\boxed{{}}.""",
    "aime25_reasoning_tr_v1": """{question}
Lütfen adım adım akıl yürüt ve nihai cevabını \\boxed{{}} içinde ver.""",
    "matharena_aime_v1": """Put your final answer within \\boxed{{}}.
The answer is an integer between 0 and 999 inclusive.

{question}""",
    "lm_eval_gsm8k_cot_v1": """Q: {question}
A:""",
    "lm_eval_arc_v1": """Question: {question}
Answer:""",
    "english_mc_cot_strict_v1": """Answer the following multiple-choice question step by step. The last line of your response must be of the form `Answer: X`, where X is the letter of the correct option.

Question: {question}

Choices:
{choices_lines}
""",
    "english_math_cot_strict_v1": """Solve the following math problem step by step. The last line of your response must be of the form `Answer: ...`, with only the final answer on that line.

Problem: {question}""",
    "english_mc_letter_v1": """Question: {question}
Choices:
{choices_lines}
Answer:""",
    "gpqa_zero_shot_v1": """What is the correct answer to this question: {question}

Choices:
{choices_paren}

Format your response as follows: "The correct answer is (insert answer here)\"""",
    "gpqa_5shot_v1": """Question: {question}
Choices:
{choices_paren}
When you're ready to answer, please use the format "The correct answer is (insert answer here).""",
    "gpqa_cot_5shot_v1": """Question: {question}
Choices:
{choices_paren}
Give step by step reasoning before you answer, and when you're ready to answer, please use the format "The correct answer is (insert answer here)":
""",
    "simple_evals_mc_v1": """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

{choices_lines}""",
    "simple_evals_mc_tr_v1": """Aşağıdaki çoktan seçmeli soruyu yanıtla. Yanıtının son satırı, HARF'in A, B, C veya D harflerinden biri olduğu 'Yanıt: $HARF' (tırnak işaretleri olmadan) biçiminde olmalıdır. Yanıtlamadan önce adım adım düşün.

{question}

{choices_lines}""",
    "simple_evals_humaneval_v1": """Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
{question}""",
}
GENERATION_PROMPTS = dict(BUILTIN_GENERATION_PROMPTS)
PROMPT_FIELDS = frozenset({"question", "choices_block", "choices_lines", "choices_paren"})
CHOICE_FIELDS = frozenset({"choices_block", "choices_lines", "choices_paren"})

FEW_SHOT_LAYOUTS = {
    "gpqa_5shot_v1": {
        "header": (
            "Here are some example questions from experts. Answer the final question "
            "yourself, following the format of the previous questions exactly.\n"
        ),
        "demonstration": (
            "Question: {question}\nChoices:\n{choices_paren}\n"
            "The correct answer is ({answer_label})\n"
        ),
    },
    "gpqa_cot_5shot_v1": {
        "header": (
            "Here are some example questions from experts. An explanation is given "
            "before the final answer. Answer the final question yourself, giving your "
            "reasoning beforehand.\n"
        ),
        "demonstration": (
            "Question: {question}\nChoices:\n{choices_paren}\n"
            "Let's think step by step: \n{explanation}\n"
            "The correct answer is ({answer_label})\n"
        ),
    },
}


def load_generation_prompts_from_registry(registry: dict) -> None:
    configured = registry.get("generation_prompts")
    if configured is None:
        return
    GENERATION_PROMPTS.clear()
    for name, value in configured.items():
        if isinstance(value, str):
            GENERATION_PROMPTS[name] = value
        else:
            GENERATION_PROMPTS[name] = value["template"]


def format_choices(example: Example) -> str:
    return "\n".join(f"{choice.label}) {choice.text}" for choice in example.choices)


def format_choices_paren(example: Example) -> str:
    return "\n".join(f"({choice.label}) {choice.text}" for choice in example.choices)


def _choice_fields(example: Any) -> dict[str, str]:
    if not getattr(example, "choices", None):
        return {"choices_lines": "", "choices_paren": ""}
    return {
        "choices_lines": format_choices(example),
        "choices_paren": format_choices_paren(example),
    }


def prompt_template_source(prompt_name: str) -> str | None:
    """Everything that determines the rendered prompt, for fingerprints."""
    template = GENERATION_PROMPTS.get(prompt_name)
    layout = FEW_SHOT_LAYOUTS.get(prompt_name)
    if template is None or layout is None:
        return template
    return template + "\n\0few_shot_layout\0" + layout["header"] + "\0" + layout["demonstration"]


def example_question(example: Any) -> str:
    return str(getattr(example, "question", getattr(example, "prompt", "")))


def example_answer(example: Any, score_target: str = "text") -> str:
    answer_label = getattr(example, "answer_label", "")
    answer_text = getattr(example, "answer_text", getattr(example, "answer", ""))
    raw = getattr(example, "raw", None)
    metadata = raw.get("metadata") if isinstance(raw, dict) else None
    demonstration_answer = (
        metadata.get("demonstration_answer") if isinstance(metadata, dict) else None
    )
    if demonstration_answer:
        return str(demonstration_answer)
    if answer_label and score_target == "label":
        return str(answer_label)
    return str(answer_text)


FEW_SHOT_RENDER_VERSION = 2
_ANSWER_MARKERS = ("Cevap:", "Answer:")


def split_prompt_template(template: str) -> tuple[str, str]:
    """(instruction, question body) of a prompt template.

    Built-in templates put a one-paragraph instruction before the question
    block; a template without a blank line is all body.
    """
    head, separator, tail = template.partition("\n\n")
    if separator and "{question}" in tail and "{question}" not in head:
        return head, tail
    return "", template


def demonstration_template(body: str) -> str:
    """The body as used for a demonstration: cut right after the answer marker,
    so a forced continuation such as "Cevap: Adım adım düşünelim." is supplied
    by the demonstration's own answer instead of being duplicated."""
    position = max(body.rfind(marker) + len(marker) if marker in body else -1 for marker in _ANSWER_MARKERS)
    return body[:position] if position > 0 else body


def format_few_shot_example(
    example: Any,
    index: int,
    append_choices: bool = False,
    score_target: str = "text",
) -> str:
    parts = [f"Örnek {index}", f"Soru: {example_question(example)}"]
    choices = getattr(example, "choices", None)
    if append_choices and choices:
        parts.append(f"Seçenekler:\n{format_choices(example)}")
    parts.append(f"Cevap: {example_answer(example, score_target=score_target)}")
    return "\n".join(parts)


def format_few_shot_block(
    examples: list[Any] | None,
    append_choices: bool = False,
    score_target: str = "text",
) -> str:
    if not examples:
        return ""
    formatted = [
        format_few_shot_example(
            example,
            index=index,
            append_choices=append_choices,
            score_target=score_target,
        )
        for index, example in enumerate(examples, start=1)
    ]
    return "Aşağıdaki örnekleri takip et.\n\n" + "\n\n".join(formatted) + "\n\n"


def build_generation_prompt(
    example: Example,
    prompt_name: str,
    append_choices: bool = False,
    few_shot_examples: list[Any] | None = None,
    score_target: str = "text",
) -> str:
    template = GENERATION_PROMPTS.get(prompt_name)
    if template is None:
        available = ", ".join(GENERATION_PROMPTS)
        raise ValueError(f"Unknown generation prompt: {prompt_name}. Available: {available}")

    choices_block = ""
    if append_choices:
        if prompt_name == "none":
            choices_block = f"\n\n{format_choices(example)}"
        else:
            choices_block = f"\nSeçenekler:\n{format_choices(example)}\n"

    layout = FEW_SHOT_LAYOUTS.get(prompt_name)
    if not few_shot_examples or layout is not None:
        target = template.format(
            question=example.question,
            choices_block=choices_block,
            **_choice_fields(example),
        )
        if not few_shot_examples:
            return target
        demonstrations = []
        for demonstration in few_shot_examples:
            raw = getattr(demonstration, "raw", None) or {}
            demonstrations.append(
                layout["demonstration"].format(
                    question=example_question(demonstration),
                    answer_label=getattr(demonstration, "answer_label", ""),
                    answer_text=getattr(demonstration, "answer_text", ""),
                    explanation=str(raw.get("explanation", "")),
                    **_choice_fields(demonstration),
                )
            )
        return layout["header"] + "".join(demonstrations) + target

    instruction, body = split_prompt_template(template)
    demonstration_body = demonstration_template(body)
    blocks = []
    for demonstration in few_shot_examples:
        demonstration_choices = ""
        if append_choices and getattr(demonstration, "choices", None):
            if prompt_name == "none":
                demonstration_choices = f"\n\n{format_choices(demonstration)}"
            else:
                demonstration_choices = f"\nSeçenekler:\n{format_choices(demonstration)}\n"
        rendered = demonstration_body.format(
            question=example_question(demonstration),
            choices_block=demonstration_choices,
            **_choice_fields(demonstration),
        )
        answer = example_answer(demonstration, score_target=score_target)
        blocks.append(f"{rendered} {answer}" if rendered and not rendered[-1].isspace() else f"{rendered}{answer}")
    blocks.append(
        body.format(
            question=example.question,
            choices_block=choices_block,
            **_choice_fields(example),
        )
    )
    prefix = f"{instruction}\n\n" if instruction else ""
    return prefix + "\n\n".join(blocks)
