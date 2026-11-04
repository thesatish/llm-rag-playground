def create_state(user_request, max_steps=5):

    return {
        "user_request": user_request,
        "tasks": [],
        "current_task": None,
        "current_step": 0,
        "max_steps": max_steps,
        "actions": [],
        "observations": [],
        "results": [],
        "finished": False,
        "final_answer": ""
    }


def add_task(state, task):

    state["tasks"].append({
        "task": task,
        "status": "pending"
    })


def complete_task(state, task):

    # Normalize the task received from the planner.
    # This makes the comparison tolerant of:
    # "Generate password"
    # "generate password"
    # "generate_password"

    normalized_task = (
        task
        .strip()
        .lower()
        .replace("_", " ")
    )

    for item in state["tasks"]:

        normalized_existing_task = (
            item["task"]
            .strip()
            .lower()
            .replace("_", " ")
        )

        if normalized_existing_task == normalized_task:

            item["status"] = "completed"

            return True

    return False


def add_action(state, action):

    state["actions"].append(action)


def record_observation(
    state,
    action,
    observation
):

    state["observations"].append({
        "step": state["current_step"],
        "action": action,
        "observation": observation
    })


def add_result(state, result):

    state["results"].append(result)


def get_pending_tasks(state):

    return [
        item["task"]
        for item in state["tasks"]
        if item["status"] == "pending"
    ]


def all_tasks_completed(state):

    if not state["tasks"]:
        return False

    return all(
        item["status"] == "completed"
        for item in state["tasks"]
    )


def next_step(state):

    state["current_step"] += 1


def can_continue(state):

    return (
        not state["finished"]
        and state["current_step"] < state["max_steps"]
    )


def finish(state, answer):

    state["final_answer"] = answer
    state["finished"] = True