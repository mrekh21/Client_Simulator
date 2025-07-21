from client_simulator import run_simulation
from email_handler import get_gmail_service
import time

if __name__ == "__main__":
    print("Starting the Client Simulator Application...")

    # Initialize Gmail Service once
    gmail_service = get_gmail_service()
    if not gmail_service:
        print("Failed to initialize Gmail service. Exiting.")
        exit()

    num_simulations = 1 # Change this to run multiple simulations in parallel or sequentially

    for i in range(num_simulations):
        print(f"\n--- Running Simulation {i + 1} of {num_simulations} ---")
        run_simulation(i + 1, gmail_service)
        if i < num_simulations - 1:
            print("\nWaiting before starting the next simulation to avoid rate limits...")
            time.sleep(30)

    print("\nAll client simulations completed successfully!")