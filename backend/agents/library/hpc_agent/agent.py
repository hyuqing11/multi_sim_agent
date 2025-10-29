import asyncio
import textwrap
from pathlib import Path

from typing_extensions import TypedDict, Literal
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

from backend.agents.llm import get_model, settings


class HPCState(MessagesState):
    """
        Represents the state of the HPC agent's execution graph.

        Attributes:
            messages: A list of messages in the conversation.
            retry_count: The number of times a job submission has been retried.
            max_retries: The maximum number of retries allowed.
            plan_data: Structured workflow plan from planner
            dft_parameters: DFT-specific parameters from planner
            current_step_group: Current workflow step being executed
    """
    retry_count: int = 0
    max_retries: int = 3
    working_directory: str
    thread_id: str
    plan_data: dict = None
    dft_parameters: dict = None
    current_step_group: dict = None


class HPCAgent:
    """
    An agent designed to manage High-Performance Computing (HPC) job submissions.
    It uses a graph-based state machine to interact with an HPC server,
    submit jobs, and automatically handle failures with a retry mechanism.
    """

    def __init__(self, server_path):
        self.server_path = str(Path(server_path))

        mcp_config = {
            "hpc": {
                "command": "python",
                "args": [self.server_path],
                "transport": "stdio"
            }
        }
        self.mcp_client = MultiServerMCPClient(mcp_config)
        self.mcp_tools = None
        self._init_lock = asyncio.Lock()
        self.agent = self._build_graph()

    async def initialize(self):
        """"
        Initializes the agent by connecting to the MCP server, fetching the
        available tools, and compiling the execution graph.
        """
        if self.mcp_tools is not None:
            return

        async with self._init_lock:
            if self.mcp_tools is not None:
                return

            self.mcp_tools = await self.mcp_client.get_tools()
            print(f"Connected to MCP server:{self.server_path}")
            print(f"Available tools: {[t.name for t in self.mcp_tools]}")

    async def _llm_node(self, state: HPCState):
        """
        This node invokes the language model to decide the next action.
        It updates the system prompt with the current retry count before invocation.
        """
        await self.initialize()

        llm = get_model(settings.DEFAULT_MODEL)
        llm_with_tools = llm.bind_tools(self.mcp_tools)

        # Build context from plan and step information
        plan_context = ""
        current_step_group = state.get("current_step_group")
        plan_data = state.get("plan_data")
        dft_parameters = state.get("dft_parameters")

        if current_step_group:
            plan_context += "\n**## Current Workflow Step**\n"
            plan_context += f"**Task:** {current_step_group.get('description', 'Run HPC job')}\n"
            steps = current_step_group.get('steps', [])
            if steps:
                plan_context += "**Specific steps:**\n"
                for i, step in enumerate(steps, 1):
                    plan_context += f"  {i}. {step}\n"

            # Add step-specific parameters
            step_params = current_step_group.get('parameters', {})
            if step_params:
                plan_context += f"\n**Step-Specific Job Parameters:**\n"
                if 'walltime' in step_params:
                    plan_context += f"- Walltime: {step_params['walltime']}\n"
                if 'nodes' in step_params:
                    plan_context += f"- Nodes: {step_params['nodes']}\n"
                if 'cores' in step_params:
                    plan_context += f"- Cores: {step_params['cores']}\n"

        if plan_data and isinstance(plan_data, dict):
            workflow_plan = plan_data.get('workflow_plan', '')
            if workflow_plan:
                plan_context += f"\n**Overall Workflow Plan:**\n{workflow_plan}\n"

        if dft_parameters:
            plan_context += f"\n**Note:** This job is part of a DFT calculation workflow.\n"

        system_prompt_template = textwrap.dedent("""\
        You are an expert autonomous HPC job submission agent.
        Your primary goal is to successfully run a computational job by submitting it to an HPC cluster. You must operate autonomously and resolve any issues without user intervention.
        {plan_context}
        **## Operating Procedure**

        Follow this procedure methodically:
        **STEP 0: Determine if User Has Existing Config**
        First, determine the config file path. If the user provides a work_dir, the config path should be `<work_dir>/config.yaml`.
        Then check if there is an existing configuration file at that path using `check_config_exists(config_path)`.
        **IF USER HAS EXISTING CONFIG**
        1. USE `read_config` to load and validate the configuration from the same path
        2. Check if it has all required field (work_dir, job_name, command, ncores, time, etc)
        3. If valid, proceed directly to STEP 5(Submit the Job)
        4. If invalid or missing fields, note the issues and ask user to clarify OR fix them yourself if obvious
        5.  **Submit the Job:** Start by using the `submit_and_monitor` tool with the configuration file path.
        6.  **Analyze the Outcome:**
            * **On SUCCESS:** Your job is done. Report the successful outcome, including the Job ID and the path to the output files.
            * **On FAILURE:** Do not give up. Begin the diagnostic and repair loop.

        **IF USER DOES NOT HAVE CONFIG:**
        1. **Analyze Request:** Understand the user's scientific goal (e.g., "DFT relaxation," "C++ compile/run"), the application, and the system size.
        2. **Gather HPC Intelligence:** You **MUST** use the `search_policy` tool to find essential, up-to-date information from the NCSU HPC documentation. This is the most critical step.
                * **Find the Module:** First, search for the correct software module. Provide the exact module string from the documentation, not a placeholder such as `lammps`.
                * **Find the Command:** Next, you **MUST** search for the specific executable path or an example run command (e.g., "NCSU HPC lammps"). Do not guess the command or executable name (like `lmp`, `mpirun lmp`, or `vasp_std`). You must paste the full, correct command from the documentation, including the correct executable path if one is shown.
                * **Find Resource Guidelines:** Research queue limits and hardware specifications to form a hypothesis for the required resources (cores, memory, etc.).
                * **If `search_policy` returns an error or no relevant content, stop and report that you cannot continue without documentation instead of guessing.**
        3. **Synthesize a Plan:** Based on your analysis and research, determine all necessary parameters for the job.
                * Capture `job_name`, `command`, and the working directory directly.
                * Build a `job_params` dictionary containing runtime settings such as `ncores`, `time`, `mem` (in GB), `modules`, `queue`, and any other scheduler directives. Only include modules and commands confirmed by the documentation you just retrieved.
                * **You must also specify the `scheduler_type`. Since you are an assistant for the NCSU HPC, you should set this to "lsf".**
        4. **Generate Configuration:** You **MUST** call the `create_job_config` tool to write the `config.yaml` file in the work_dir. The config_path should be `<work_dir>/config.yaml`. Pass all the parameters you determined in the previous step in the `job_params` dictionary. **DO NOT** write YAML in your response; use the tool.
        5.  **Submit the Job:** Start by using the `submit_and_monitor` tool with the user-provided configuration file.
        6.  **Analyze the Outcome:**
            * **On SUCCESS:** Your job is done. Report the successful outcome, including the Job ID and the path to the output files.
            * **On FAILURE:** Do not give up. Begin the diagnostic and repair loop.

        **## Diagnostic and Repair Loop**

        If the job fails, you MUST follow these steps in order:

        1.  **Diagnose the Error:** Use `read_job_output` to carefully examine the error logs from the failed job.
        2.  **Form a Hypothesis:** Based on the error log, state your hypothesis for the root cause of the failure. For example: "The error 'invalid queue' suggests the queue name in the config is wrong."
        3.  **Gather Information & Fix:**
            * Use `read_config` to inspect the current job configuration.
            * If the error is related to resource limits, partitions, queue names, modules, or commands, you MUST use the `search_policy` tool to find the correct values from the HPC documentation.
            * If you see scheduler error like "Too few tasks requested" or "Job not submitted",
                do Not search for exact phrase
                Instead, search for the HPC policy describing:
                    - queue minimum cores or tasks
                    - job submission requirements
                Example query: "site:hpc.ncsu.edu standard queue minimum tasks"
            * Based on your hypothesis and the information gathered, formulate a corrected configuration.
            * Use `update_config` to apply the fix.
        4.  **Retry:** Use `submit_and_monitor` to try running the job again.
        5.  **Repeat:** If the job fails again, repeat this loop.

        **## State**
        You have a maximum of {max_retries} attempts.
        You are currently on attempt: {current_attempt}
        """)
        system_prompt_content = system_prompt_template.format(
            plan_context=plan_context,
            max_retries=state.get("max_retries", 3),
            current_attempt=state.get("retry_count", 0) + 1,
        )

        # Create a new messages list instead of mutating the state
        messages = list(state['messages'])
        if messages and isinstance(messages[0], SystemMessage):
            # Replace the first message with updated system prompt
            messages = [SystemMessage(content=system_prompt_content)] + messages[1:]
        else:
            # Insert system prompt at the beginning
            messages = [SystemMessage(content=system_prompt_content)] + messages

        response = await llm_with_tools.ainvoke(messages)
        # Only return fields that this node modifies
        return {
            "messages": [response]
        }

    async def _tool_node(self, state: HPCState):
        """
        This node executes the tools called by the language model.
        It processes tool calls, invokes the corresponding tool from the MCP client,
        and updates the retry count if a job submission fails.
        """
        await self.initialize()

        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls
        tools_by_name = {tool.name: tool for tool in self.mcp_tools}
        new_retry_count = state["retry_count"]
        observations = []

        for tc in tool_calls:
            tool = tools_by_name.get(tc["name"])
            if not tool:
                observation = f"Error: Tool {tc['name']} not found."
            else:
                try:
                    print(f"Executing tool: {tc['name']} with args: {tc['args']}")
                    observation = await tool.ainvoke(tc["args"])
                    if tc["name"] == 'submit_and_monitor':
                        if "FAILED" in observation or "ERROR" in observation:
                            new_retry_count += 1
                            print(f"Job failed: (attempt {new_retry_count}/{state['max_retries']})")
                        elif "SUCCESS" in observation:
                            print("Job succeeded")
                except Exception as e:
                    observation = f"Error executing tool {tc['name']}: {e}"
                    print(f"Tool execution error: {e}")
            observations.append(observation)

        tool_output = [
            ToolMessage(
                content=str(obs),
                name=tc["name"],
                tool_call_id=tc["id"]
            )
            for obs, tc in zip(observations, tool_calls)
        ]

        # Only return fields that this node modifies
        return {
            "messages": tool_output,
            "retry_count": new_retry_count
        }

    def _should_continue(self, state: HPCState) -> Literal["tools", "end"]:
        """
        This conditional edge determines the next step in the graph.
        - If the max retries have been reached, it ends the execution.
        - If the model has made a tool call, it routes to the tool node.
        - Otherwise, it ends the execution.
        """
        if state.get("retry_count", 0) >= state.get("max_retries", 3):
            print(f"Max retries reached: {state['retry_count']}")
            return "end"
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or \
                     last_message.additional_kwargs.get("tool_calls", [])
        if tool_calls:
            return "tools"
        return "end"

    def _build_graph(self):
        """
        Builds and compiles the state graph for the agent.
        """
        graph = StateGraph(HPCState)
        graph.add_node("llm", self._llm_node)
        graph.add_node("tools", self._tool_node)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm",
                                    self._should_continue,
                                    {
                                        "tools": "tools",
                                        "end": END
                                    }
                                    )
        graph.add_edge("tools", "llm")
        memory = MemorySaver()
        agent = graph.compile(checkpointer=memory)
        return agent


def create_hpc_agent():
    """Factory that returns the compiled HPC agent graph."""
    server_path = Path(__file__).with_name("hpc_server.py")
    return HPCAgent(server_path).agent


async def main():
    import uuid

    server_path = Path(__file__).with_name("hpc_server.py")
    agent = HPCAgent(server_path)
    await agent.initialize()
    print("=" * 60)
    print("HPC Job Submission Agent")
    print("=" * 60)
    print("\nAgent initialized successfully!")
    print(f"Available tools: {[t.name for t in agent.mcp_tools]}")
    print("\n" + "=" * 60)
    user_request = """
        I need to run a VASP DFT relaxation for a silicon crystal.
        My working directory is /share/mygroup/si_relax
        The structure is a 2x2x2 supercell with INCAR, POSCAR, POTCAR, and KPOINTS already set up."""

    thread_id = str(uuid.uuid4())

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=user_request)],
        "retry_count": 0,
        "max_retries": 3,
        "thread_id": thread_id
    }

    # Configuration for the agent execution
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    print("\nUser Request:")
    print("-" * 60)
    print(user_request.strip())
    print("-" * 60)
    print("\nStarting agent execution...\n")

    try:
        # Run the agent
        async for event in agent.agent.astream(initial_state, config):
            print("\n" + "=" * 60)
            print(f"Event: {list(event.keys())}")
            print("=" * 60)

            for node_name, node_output in event.items():
                print(f"\n--- Node: {node_name} ---")

                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        # Print message type and content
                        msg_type = type(msg).__name__
                        print(f"\nMessage Type: {msg_type}")

                        if hasattr(msg, 'content') and msg.content:
                            print(f"Content: {msg.content[:500]}")
                            if len(msg.content) > 500:
                                print("... (truncated)")

                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"\nTool Calls: {len(msg.tool_calls)}")
                            for tc in msg.tool_calls:
                                print(f"  - {tc['name']}")
                                print(f"    Args: {tc['args']}")

                if "retry_count" in node_output:
                    print(f"\nRetry Count: {node_output['retry_count']}/{node_output.get('max_retries', 3)}")

        print("\n" + "=" * 60)
        print("Agent execution completed!")
        print("=" * 60)

        # Get final state
        final_state = await agent.agent.aget_state(config)
        final_messages = final_state.values.get("messages", [])

        if final_messages:
            last_message = final_messages[-1]
            print("\n--- Final Result ---")
            print(f"Message Type: {type(last_message).__name__}")
            if hasattr(last_message, 'content'):
                print(f"Content:\n{last_message.content}")

    except Exception as e:
        print(f"\n❌ Error during agent execution: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup: close the MCP client connection
        print("\nClosing MCP client connection...")
        # Note: MultiServerMCPClient doesn't have an explicit close in the standard API,
        # but you might want to add cleanup if needed


if __name__ == "__main__":
    asyncio.run(main())