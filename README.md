# Client Simulator for Tour Booking

Python-based client simulator designed to automate and test communication flows between a simulated client and a tour booking platform (or a human agent representing one). It uses the Gmail API for email interaction and large language models (LLMs) to generate dynamic and context-aware client responses, mimicking a real customer conversation.

### Key Features

  * **Automated Email Communication:** Sends initial inquiries and subsequent replies and fetches responses using the Gmail API.
  * **Dynamic Response Generation:** Leverages LLMs (via LangChain/LangGraph) to generate realistic client messages based on the conversation history and predefined scenarios.
  * **Conversation State Management:** Tracks the `conversation_stage` (e.g., initial request, information gathering, draft review) to guide the client's behavior.
  * **Simulation Logging:** Records each turn of the simulation for later analysis.

### Technologies Used

  * **LangChain / LangGraph:** For building and managing the conversational workflow.
  * **Gmail API:** For sending and receiving emails programmatically.
  * **OpenAI API (or Google Gemini API):** For LLM capabilities.

### Setup and Installation

Follow these steps to get the client simulator up and running:

1.  **Clone the Repository (or Initialize)**
    If you've already initialized your local repository, you can skip `git clone`. Otherwise:

    ```bash
    git clone https://github.com/mrekh21/Client_Simulator.git
    cd Client_Simulator
    ```

2.  **Create a Virtual Environment**
    It's best practice to use a virtual environment to manage project dependencies:

    ```bash
    python -m venv .venv
    ```

    Activate the virtual environment:

      * **Windows:** `.\.venv\Scripts\activate`
      * **macOS/Linux:** `source ./.venv/bin/activate`

3.  **Install Dependencies**
    Install all required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Gmail API Credentials**
    The simulator uses the Gmail API to send and receive emails.

      * Go to the Google Cloud Console: [https://console.cloud.google.com/](https://console.cloud.google.com/)
      * Create a new project or select an existing one.
      * Enable the **Gmail API** for your project.
      * Go to "Credentials," click "Create Credentials," and select "OAuth client ID."
      * Choose "Desktop app" as the application type.
      * Download the `credentials.json` file. Place this file in the **root directory** of your `Client_Simulator` project.
      * The first time you run `main.py`, a browser window will open to authorize the application. After successful authorization, a `token.json` file will be automatically created in your project root, storing your authentication tokens.

5.  **Configure Environment Variables (.env file)**
    Create a file named `.env` in the **root directory** of your project (where `main.py` is located). Add the following variables, replacing the placeholders with your actual values:

    ```env
    # Your Gmail account to send/receive emails
    SIMULATOR_EMAIL_ADDRESS=your_client_email@gmail.com
    PLATFORM_EMAIL_ADDRESS = "your_company_email@gmail.com"

    # Your OpenAI API key/Google API Key
    GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
    OPENAI_API_KEY=YOUR_OPENAI_API_KEY
    ```

### How to Run the Simulation

Once all dependencies and credentials are set up, you can run the simulator:

```bash
python main.py
```

The application will start, initialize the Gmail service, and begin the client simulation. Follow the console output to track the conversation turns.
