from state import (
    create_state,
    add_task,
    add_action,
    record_observation,
    add_result,
    complete_task,
    all_tasks_completed,
    next_step,
    can_continue,
    finish
)

from task_planner import (
    decompose_task
)


def parse_decision(decision):

    task = None
    tool = None

    # Handle the normal two-line format
    for line in decision.splitlines():

        line = line.strip()

        if line.startswith("TASK:"):

            task = line[5:].strip()

        elif line.startswith("TOOL:"):

            tool = line[5:].strip()

    # Handle the case where the LLM puts
    # TASK and TOOL on the same line.
    if task and "TOOL:" in task:

        parts = task.split(
            "TOOL:",
            1
        )

        task = parts[0].strip()

        if not tool:

            tool = parts[1].strip()

    return task, tool




async def run_agent_loop(
    user_request,
    planner,
    execute_action,
    format_answer,
    tools,
    client
):

    state = create_state(
        user_request
    )

    # --------------------------------
    # Step 1: Decompose the goal
    # --------------------------------

    planned_tasks = decompose_task(
        user_request
    )

    for task in planned_tasks:

        add_task(
            state,
            task
        )

    print("\nTask Plan")
    print("---------")

    for item in state["tasks"]:

        print(
            "-",
            item["task"]
        )

    # --------------------------------
    # Step 2: Autonomous loop
    # --------------------------------

    while can_continue(state):

        # --------------------------------
        # Check whether all tasks are done
        # --------------------------------

        if all_tasks_completed(state):

            print(
                "\n--- Step "
                f"{state['current_step'] + 1} ---"
            )

            print(
                "Planner Decision:"
            )

            print("TASK: FINISH")

            answer = format_answer(
                state
            )

            finish(
                state,
                answer
            )

            break

        print(
            f"\n--- Step "
            f"{state['current_step'] + 1} ---"
        )

        # --------------------------------
        # Ask planner for next task
        # --------------------------------

        decision = planner(
            state,
            tools
        )

        print(
            "Planner Decision:"
        )

        print(decision)

        # --------------------------------
        # Check FINISH
        # --------------------------------

        if decision.strip().upper() == "FINISH":

            answer = format_answer(
                state
            )

            finish(
                state,
                answer
            )

            break

        # --------------------------------
        # Extract task and tool
        # --------------------------------

        task, tool = parse_decision(
            decision
        )

        if not task or not tool:

            add_result(
                state,
                "Invalid planner decision."
            )

            next_step(state)

            continue

        state["current_task"] = task

        # --------------------------------
        # Execute tool
        # --------------------------------

        observation = await execute_action(
            tool,
            state,
            client
        )

        print(
            "Observation:",
            observation
        )

        # --------------------------------
        # Update state
        # --------------------------------

        add_action(
            state,
            tool
        )

        record_observation(
            state,
            tool,
            observation
        )

        add_result(
            state,
            observation
        )

        # --------------------------------
        # Mark task complete
        # --------------------------------

        if not observation.lower().startswith(
            "error"
        ):

            complete_task(
                state,
                task
            )

            print(
                "completed"
            )

        else:

            add_result(
                state,
                "The previous action failed. "
                "The planner must reconsider."
            )

        next_step(state)

    return state