import asyncio
from argparse import ArgumentParser
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.agents.library.orchestrator.agent import orchestrator_agent


async def _run(prompt: str) -> None:
    config = RunnableConfig(configurable={"thread_id": f"orchestrator-demo-{uuid4()}"})
    state = {"messages": [HumanMessage(content=prompt)]}
    result = await orchestrator_agent.ainvoke(state, config=config)

    print("=== Orchestrator Output ===")
    for message in result.get("messages", []):
        role = getattr(message, "type", message.__class__.__name__)
        content = getattr(message, "content", str(message))
        print(f"[{role}]\n{content}\n")


def main() -> None:
    parser = ArgumentParser(description="Quick smoke-test driver for the orchestrator agent.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Plan a DFT workflow to relax silicon and report any errors.",
        help="User request forwarded to the orchestrator supervisor.",
    )
    args = parser.parse_args()
    query = 'lattice constant of YH2'
    asyncio.run(_run(args.prompt))


if __name__ == "__main__":
    main()
