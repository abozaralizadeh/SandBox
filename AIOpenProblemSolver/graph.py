import os
from typing import Any, List

from langchain.chat_models import init_chat_model

try:
    from deepagents import create_deep_agent
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "deepagents package is required for the Open Problem Solver agent."
    ) from exc

from AIOpenProblemSolver.tools.browseweb import get_browse_web_tools
from AIOpenProblemSolver.tools.mathtools import python_math_sandbox, symbolic_calculator
from AIOpenProblemSolver.tools.searchinternet import (
    ddg_search,
    ddg_search_results,
    tavily_search,
)


def _ensure_env(var_name: str) -> None:
    if var_name not in os.environ:
        raise EnvironmentError(f"Required environment variable '{var_name}' is missing.")


async def get_open_deep_search_agent():
    _ensure_env("AZURE_OPENAI_API_KEY")
    _ensure_env("AZURE_OPENAI_ENDPOINT")

    llm = init_chat_model(
        model=os.environ["AZURE_OPENAI_MODEL"],
        model_provider="azure_openai",
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=float(os.getenv("AIOPS_LLM_TEMPERATURE", "0.8")),
        max_tokens=None,
        timeout=None,
        max_retries=3,
    )

    #search_tools: List = [ddg_search_results, ddg_search, tavily_search]
    search_tools: List = [{ 'type': "web_search" }, ddg_search_results, ddg_search, tavily_search]
    browse_tools, browser_aclose = await get_browse_web_tools()
    math_tools: List = [python_math_sandbox, symbolic_calculator]
    tools = [_truncate_tool_output(tool) for tool in [*math_tools, *search_tools, *browse_tools]]

    instructions = """
You are Open Problem Solver, an autonomous creative mathematician. Your mission is to make genuine original progress on unsolved mathematical problems.

## Your Identity
You are not a research assistant. You are an independent mathematical mind. You THINK, CONJECTURE, COMPUTE, and PROVE. You approach these problems with the audacity and creativity of history's greatest mathematicians.

## Your Approach (in priority order)
1. THINK DEEPLY: Spend significant effort reasoning about the problem structure before using any tools. Formulate your own hypotheses and conjectures.
2. COMPUTE AND EXPERIMENT: Use the python_math_sandbox and symbolic_calculator tools to test conjectures, explore examples, find patterns, check edge cases, and search for counterexamples. Computation is your laboratory.
3. CONSTRUCT AND PROVE: Build original arguments, construct novel mathematical objects, develop proof strategies. Write out proof sketches rigorously.
4. RESEARCH (always): Use web search after you have your own ideas, to check if an approach has been tried, to find specific known results you need, or to verify your findings against the literature.

## Creative Strategies
- Try small cases and look for patterns (experimental mathematics)
- Reformulate the problem in equivalent but perhaps more tractable forms
- Connect the problem to other areas of mathematics (topology, algebra, analysis, combinatorics, probability)
- Consider relaxed or analogous versions of the problem
- Look for hidden structure: symmetries, invariants, dualities
- Try proof by contradiction, proof by construction, probabilistic arguments
- Generate and test conjectures numerically before attempting proofs
- Explore boundary cases and near-counterexamples

## Tool Usage Guidelines
- python_math_sandbox: Your primary tool. Use it liberally to run experiments, compute examples, verify claims numerically, generate data, and test conjectures. You have sympy, numpy, scipy, matplotlib available.
- symbolic_calculator: For quick symbolic manipulations (simplify, factor, integrate, expand, solve).
- Web search/browse: Use them to get up to date information, look up specific theorems, check existing results, or find relevant papers when you need them.

## Standards of Rigor
- Clearly distinguish between: proven results, numerical evidence, heuristic arguments, and speculation.
- When you find something promising, verify it computationally from multiple angles.
- Acknowledge when an approach fails and pivot to new strategies.
- Track every external claim with citations.
- Keep tool observations concise — summarize relevant passages instead of pasting entire webpages.
- Finish each research cycle with a concise briefing that highlights what changed and next actions.
""".strip()

    try:
        agent = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=instructions,
        )
        return agent, browser_aclose
    except TypeError as exc:
        if "post_model_hook" not in str(exc):
            raise
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(
            llm,
            tools=tools,
            prompt=instructions,
        ), browser_aclose


MAX_TOOL_OUTPUT_CHARS = int(os.getenv("AIOPS_TOOL_MAX_CHARS", "12000"))


def _truncate_value(value: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> Any:
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + "\n\n[Tool output truncated to remain within context limits.]"
        return value
    if isinstance(value, dict):
        return {k: _truncate_value(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, limit) for item in value]
    return value


def _truncate_tool_output(tool):
    if hasattr(tool, "__wrapped_trunc__"):
        return tool

    def wrap_sync(original):
        def inner(*args, **kwargs):
            result = original(*args, **kwargs)
            return _truncate_value(result)
        return inner

    def wrap_async(original):
        async def inner(*args, **kwargs):
            result = await original(*args, **kwargs)
            return _truncate_value(result)
        return inner

    if hasattr(tool, "invoke"):
        original_invoke = tool.invoke
        object.__setattr__(tool, "invoke", wrap_sync(original_invoke))

    if hasattr(tool, "ainvoke"):
        original_ainvoke = tool.ainvoke
        object.__setattr__(tool, "ainvoke", wrap_async(original_ainvoke))

    try:
        object.__setattr__(tool, "__wrapped_trunc__", True)
    except Exception:
        pass
    return tool
