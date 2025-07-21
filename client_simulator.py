import time
import json
import os
from datetime import datetime

from agent_graph import build_graph, AgentState
from email_handler import get_gmail_service
from config import (
    COMPANY_INFO, PLATFORM_EMAIL_ADDRESS, SIMULATOR_EMAIL_ADDRESS,
    MAX_CONVERSATION_TURNS, LOG_FILE_PATH
)


def run_simulation(simulation_id: int, gmail_service):
    """Runs a single client simulation."""
    print(f"\n--- Starting Simulation {simulation_id} ---")

    initial_state = AgentState(
        chat_history=[],
        current_request={},
        platform_draft=None,
        conversation_stage='start',  # Custom stage to trigger initial_request
        num_modifications_requested=0,
        is_satisfied=False,
        is_booking_finalized=False,
        company_info=COMPANY_INFO,
        last_email_sent_time=None
    )

    app = build_graph(gmail_service)

    log_data = {
        "simulation_id": simulation_id,
        "start_time": datetime.now().isoformat(),
        "conversation_log": [],
        "final_status": "in_progress",
        "end_time": None,
        "total_turns": 0
    }

    try:
        # Run the graph
        for i, s in enumerate(app.stream(initial_state), 1):
            if i > MAX_CONVERSATION_TURNS:
                print(
                    f"Simulation {simulation_id}: Reached max conversation turns ({MAX_CONVERSATION_TURNS}). Stopping.")
                log_data["final_status"] = "max_turns_reached"
                break

            # Print current state
            print(f"\n--- Simulation {simulation_id} - Turn {i} ---")
            current_node = list(s.keys())[-1]  # Get the last node executed
            print(f"Current Node: {current_node}")

            # Update state with the changes from the current node
            initial_state.update(s[current_node])

            # Log conversation turn
            log_data["conversation_log"].append({
                "turn": i,
                "node": current_node,
                "chat_history_snapshot": initial_state['chat_history'][-1] if initial_state['chat_history'] else None,
                "stage": initial_state['conversation_stage'],
                "is_satisfied": initial_state['is_satisfied'],
                "is_booking_finalized": initial_state['is_booking_finalized'],
                "timestamp": datetime.now().isoformat()
            })
            log_data["total_turns"] = i

            if initial_state['is_booking_finalized']:
                print(f"Simulation {simulation_id}: Booking finalized!")
                log_data["final_status"] = "finalized"
                break

    except Exception as e:
        print(f"Simulation {simulation_id} encountered an error: {e}")
        log_data["final_status"] = f"error: {e}"
    finally:
        log_data["end_time"] = datetime.now().isoformat()
        save_log(log_data)
        print(f"--- Simulation {simulation_id} Ended ({log_data['final_status']}) ---")


def save_log(log_data):
    """Saves simulation log data to a JSON file."""
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    with open(LOG_FILE_PATH, 'a') as f:
        f.write(json.dumps(log_data) + "\n")
    print(f"Simulation log saved to {LOG_FILE_PATH}")

