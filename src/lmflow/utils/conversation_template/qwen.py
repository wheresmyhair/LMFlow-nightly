#!/usr/bin/env python
# Copyright 2024 Statistics and Machine Learning Research Group. All rights reserved.


from .base import ConversationTemplate, ConversationTemplateForTool, StringFormatter, TemplateComponent

QWEN2_TEMPLATE = ConversationTemplate(
    template_name="qwen2",
    user_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>user\n{{content}}<|im_end|>\n")]
    ),
    assistant_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>assistant\n{{content}}<|im_end|>\n")]
    ),
    system_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>system\n{{content}}<|im_end|>\n")]
    ),
    separator=TemplateComponent(type="string", content="\n"),
)


QWEN2_TEMPLATE_FOR_TOOL = ConversationTemplateForTool(
    template_name="qwen2_for_tool",
    user_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>user\n{{content}}<|im_end|>\n")]
    ),
    function_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>assistant\n{{content}}<|im_end|>\n")]
    ),
    observation_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>tool\n{{content}}<|im_end|>\n")]
    ),
    assistant_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>assistant\n{{content}}<|im_end|>\n")]
    ),
    system_formatter=StringFormatter(
        template=[TemplateComponent(type="string", content="<|im_start|>system\n{{content}}<|im_end|>\n")]
    ),
    separator=TemplateComponent(type="string", content="\n"),
)


QWEN2_5_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}"
    "{%- endif %}"
    '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
    '{%- if message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}"
    "{{ message.content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    '{%- elif message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}"
    "{% generation %}"
    "{{- '\\n' + message.content }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{% generation %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{{- tool_call.arguments | tojson }}"
    "{{- '}\\n</tool_call>' }}"
    "{% endgeneration %}"
    "{%- endfor %}"
    "{% generation %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


QWEN2_5_1M_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'You are a helpful assistant.' }}"
    "{%- endif %}"
    '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
    '{%- if message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}"
    "{{ message.content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    '{%- elif message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}"
    "{% generation %}"
    "{{- '\\n' + message.content }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{% generation %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{{- tool_call.arguments | tojson }}"
    "{{- '}\\n</tool_call>' }}"
    "{% endgeneration %}"
    "{%- endfor %}"
    "{% generation %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


QWEN2_5_MATH_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'Please reason step by step, and put your final answer within \\\\boxed{}.' }}"
    "{%- endif %}"
    '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nPlease reason step by step, and put your final answer within \\\\boxed{}.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
    '{%- if message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}"
    "{{ message.content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    '{%- elif message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}"
    "{% generation %}"
    "{{- '\\n' + message.content }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{% generation %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{{- tool_call.arguments | tojson }}"
    "{{- '}\\n</tool_call>' }}"
    "{% endgeneration %}"
    "{%- endfor %}"
    "{% generation %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


QWEN_QWQ_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- messages[0]['content'] }}"
    "{%- else %}"
    "{{- 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.' }}"
    "{%- endif %}"
    '{{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    '{%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}'
    '{%- if message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{% generation %}"
    "{{ message.content + '<|im_end|>' + '\\n' }}"
    "{% endgeneration %}"
    "{%- else %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    '{%- elif message.role == "assistant" %}'
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}"
    "{% generation %}"
    "{{- '\\n' + message.content }}"
    "{% endgeneration %}"
    "{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    "{% generation %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{{- tool_call.arguments | tojson }}"
    "{{- '}\\n</tool_call>' }}"
    "{% endgeneration %}"
    "{%- endfor %}"
    "{% generation %}"
    "{{- '<|im_end|>\\n' }}"
    "{% endgeneration %}"
    '{%- elif message.role == "tool" %}'
    '{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

# Wraps an assistant message's generated part in a generation span unless the message sets
# `loss: false`. Message-level loss relies on this macro; templates without it reject `loss`.
_MAYBE_GENERATION_MACRO = (
    "{%- macro maybe_generation(enabled) -%}"
    "{%- set body = caller() -%}"
    "{%- if enabled -%}"
    "{% generation %}{{- body }}{% endgeneration %}"
    "{%- else -%}"
    "{{- body }}"
    "{%- endif -%}"
    "{%- endmacro -%}"
)

_QWEN3_SYSTEM_AND_TOOLS = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0].role == 'system' %}"
    "{{- messages[0].content + '\\n\\n' }}"
    "{%- endif %}"
    '{{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}'
    "{%- for tool in tools %}"
    '{{- "\\n" }}'
    "{{- tool | tojson }}"
    "{%- endfor %}"
    '{{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}'
    "{%- else %}"
    "{%- if messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
)

_QWEN3_TOOL_CALLS = (
    "{%- if message.tool_calls %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if (loop.first and content) or (not loop.first) %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- if tool_call.function %}"
    "{%- set tool_call = tool_call.function %}"
    "{%- endif %}"
    '{{- \'<tool_call>\\n{"name": "\' }}'
    "{{- tool_call.name }}"
    '{{- \'", "arguments": \' }}'
    "{%- if tool_call.arguments is string %}"
    "{{- tool_call.arguments }}"
    "{%- else %}"
    "{{- tool_call.arguments | tojson }}"
    "{%- endif %}"
    "{{- '}\\n</tool_call>' }}"
    "{%- endfor %}"
    "{%- endif %}"
)

_QWEN3_TOOL_RESPONSE = (
    '{%- elif message.role == "tool" %}'
    '{%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}'
    "{{- '<|im_start|>user' }}"
    "{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}"
    "{{- message.content }}"
    "{{- '\\n</tool_response>' }}"
    '{%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}'
    "{{- '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
)

# Qwen3's own chat template, with generation spans for assistant-only loss. The span covers what the
# model generates after the `<|im_start|>assistant\n` prompt, up to and including `<|im_end|>`.
# Like the original, it drops reasoning from assistant turns before the last user query and adds an
# empty think block to the last turn, so multi-turn data whose observations arrive as user messages
# is rewritten; the train/inference prefix check reports that. Use `qwen3_nothink` for such data.
QWEN3_TEMPLATE = (
    _MAYBE_GENERATION_MACRO
    + _QWEN3_SYSTEM_AND_TOOLS
    + (
        "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}"
        "{%- for message in messages[::-1] %}"
        "{%- set index = (messages|length - 1) - loop.index0 %}"
        "{%- if ns.multi_step_tool and message.role == \"user\" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}"
        "{%- set ns.multi_step_tool = false %}"
        "{%- set ns.last_query_index = index %}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- for message in messages %}"
        '{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}'
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
        '{%- elif message.role == "assistant" %}'
        "{{- '<|im_start|>' + message.role + '\\n' }}"
        "{%- call maybe_generation(message.loss is not defined or message.loss is none or message.loss) -%}"
        "{%- set content = message.content %}"
        "{%- set reasoning_content = '' %}"
        "{%- if message.reasoning_content is defined and message.reasoning_content is not none %}"
        "{%- set reasoning_content = message.reasoning_content %}"
        "{%- else %}"
        "{%- if '</think>' in message.content %}"
        "{%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}"
        "{%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}"
        "{%- endif %}"
        "{%- endif %}"
        "{%- if loop.index0 > ns.last_query_index %}"
        "{%- if loop.last or (not loop.last and reasoning_content) %}"
        "{{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
        "{%- else %}"
        "{{- content }}"
        "{%- endif %}"
        "{%- else %}"
        "{{- content }}"
        "{%- endif %}"
    )
    + _QWEN3_TOOL_CALLS
    + "{{- '<|im_end|>' }}{%- endcall -%}{{- '\\n' }}"
    + _QWEN3_TOOL_RESPONSE
    + (
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- if enable_thinking is defined and enable_thinking is false %}"
        "{{- '<think>\\n\\n</think>\\n\\n' }}"
        "{%- endif %}"
        "{%- endif %}"
    )
)

# Qwen3 rendered without thinking and without rewriting history, equivalent to LLaMA-Factory's
# `qwen3_nothink`: every assistant message is rendered verbatim, with no think block, and the
# generation prompt is `<|im_start|>assistant\n`. With it, the training context of every assistant
# turn equals the inference prompt for that turn, which suits agent trajectories that return
# observations as user messages (e.g. OpenHands with non-native function calling). Messages that
# carry `reasoning_content` are rejected, since this template has no place to render it.
QWEN3_NOTHINK_TEMPLATE = (
    _MAYBE_GENERATION_MACRO
    + _QWEN3_SYSTEM_AND_TOOLS
    + (
        "{%- for message in messages %}"
        '{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}'
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
        '{%- elif message.role == "assistant" %}'
        "{%- if message.reasoning_content %}"
        "{{- raise_exception('qwen3_nothink renders assistant messages verbatim and has no place for reasoning_content') }}"
        "{%- endif %}"
        "{{- '<|im_start|>' + message.role + '\\n' }}"
        "{%- call maybe_generation(message.loss is not defined or message.loss is none or message.loss) -%}"
        "{%- set content = message.content if message.content is string else '' %}"
        "{{- content }}"
    )
    + _QWEN3_TOOL_CALLS
    + "{{- '<|im_end|>' }}{%- endcall -%}{{- '\\n' }}"
    + _QWEN3_TOOL_RESPONSE
    + "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
