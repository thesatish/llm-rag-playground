from state import (
    create_state,
    add_action,
    record_observation,
    next_step,
    can_continue,
    finish
)


async def run_agent_loop(
    user_request,
    planner,
    execute_action,
    format_answer,
    tools,
    client
):

    state = create_state(user_request)
# While can_continue: "This is the heart of our agent.“

    while can_continue(state):
#The planner decides what to do next.

        action = planner(state, tools)

        print("Planner Selected:", action)

        if action == "FINISH":

            answer = format_answer(state)

            finish(state, answer)

            break
# We execute that action.

        observation = await execute_action(
            action,
            client
        )
# We add that action in state

        add_action(
            state,
            action
        )
#    	We remember what happened.

        record_observation(
            state,
            action,
            observation
        )
#	We move to the next iteration.
     

        next_step(state)

    return state
