def create_state(user_request, max_steps=5):
    return {
        "user_request": user_request,
        "current_step": 0,
        "max_steps": max_steps,
        "actions": [],
        "observations": [],
        "finished": False,
        "final_answer": ""
    }

def add_action(state, action):
    state["actions"].append(action)

def record_observation(state, action, observation):
    state["observations"].append(
        {
            "step": state["current_step"],
            "action": action,
            "observation": observation
        }
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
