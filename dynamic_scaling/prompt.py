"""Prompts for generating data."""

# For generating the final dataset
SKY_T1_FIXED = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: [begin_of_thought] {thought with steps separated with '\\n\\n'} [end_of_thought] Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: [begin_of_solution] {final formatted, precise, and clear solution} [end_of_solution] Now, try to solve the following question through the above guidelines:"  # noqa


BASE_MODEL_SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within [begin_of_thought] [begin_of_thought] and [begin_of_solution] [begin_of_solution] tags, respectively, i.e., [begin_of_thought] reasoning process here [end_of_thought] [begin_of_solution] answer here [end_of_solution]. User:"""

# For data generation using deepseek-r1
SKY_T1_SYSTEM_PROMPT = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process
before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of
analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered
thinking process.
"""


def generate_prompt(test_case, prompt, starter_code=None):
    """Generate a prompt for the LLM to solve a problem."""
    formatted_prompt = ""

    data = test_case
    if not data.get("fn_name"):
        formatted_prompt += "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."  # noqa
    else:
        formatted_prompt += (
            "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."  # noqa
        )

    data = prompt
    formatted_prompt += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data  # + "\n"
        formatted_prompt += data

    return formatted_prompt